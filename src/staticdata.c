// This file is a part of Julia. License is MIT: https://julialang.org/license

/*
  saving and restoring system images

  This performs serialization and deserialization of system and package images. It creates and saves a compact binary
  blob, making deserialization "simple" and fast: we "only" need to deal with uniquing, pointer relocation,
  method root insertion, registering with the garbage collector, making note of special internal types, and
  backedges/invalidation. Special objects include things like builtin functions, C-implemented types (those in jltypes.c),
  the metadata for documentation, optimal layouts, integration with native system image generation, and preparing other
  preprocessing directives.

  During serialization, the flow has several steps:

  - step 1 inserts relevant items into `serialization_order`, an `obj` => `id::Int` mapping. `id` is assigned by
    order of insertion. This stage is implemented by `jl_queue_for_serialization` and its callees;
    while it would be simplest to use recursion, this risks stack overflow, so recursion is mimicked
    using a work-queue managed by `jl_serialize_reachable`.

    It's worth emphasizing that the only goal of this stage is to insert objects into `serialization_order`.
    In later stages, such objects get written in order of `id`.

  - step 2 (the biggest of four steps) takes all items in `serialization_order` and actually serializes them ordered
    by `id`. The system is serialized into several distinct streams (see `jl_serializer_state`), a "main stream"
    (the `s` field) as well as parallel streams for writing specific categories of additional internal data (e.g.,
    global data invisible to codegen, as well as deserialization "touch-up" tables, see below). These different streams
    will be concatenated in later steps. Certain key items (e.g., builtin types & functions associated with `INSERT_TAG`
    below, integers smaller than 512) get serialized via a hard-coded tag table.

    Serialization builds "touch up" tables used during deserialization. Pointers and items requiring gc
    registration get encoded as `(location, target)` pairs in `relocs_list` and `gctags_list`, respectively.
    `location` is the site that needs updating (e.g., the address of a pointer referencing an object), and is
    set to `position(s)`, the offset of the object from the beginning of the deserialized blob.
    `target` is a bitfield-encoded index into lists of different categories of data (e.g., mutable data, constant data,
    symbols, functions, etc.) to which the pointer at `location` refers. The different lists and their bitfield flags
    are given by the `RefTags` enum: if `t` is the category tag (one of the `RefTags` enums) and `i` is the index into
    one of the corresponding categorical list, then `index = t << RELOC_TAG_OFFSET + i`. The simplest source for the
    details of this encoding can be found in the pair of functions `get_reloc_for_item` and `get_item_for_reloc`.

    `uniquing` also holds the serialized location of external DataTypes, MethodInstances, and singletons
    in the serialized blob (i.e., new-at-the-time-of-serialization specializations).

    Most of step 2 is handled by `jl_write_values`, followed by special handling of the dedicated parallel streams.

  - step 3 combines the different sections (fields of `jl_serializer_state`) into one

Much of the "real work" during deserialization is done by `get_item_for_reloc`. But a few items require specific
attention:
- uniquing: during deserialization, the target item (an "external" type or MethodInstance) must be checked against
  the running system to see whether such an object already exists (i.e., whether some other previously-loaded package
  or workload has created such types/MethodInstances previously) or whether it needs to be created de-novo.
  In either case, all references at `location` must be updated to the one in the running system.
    `new_dt_objs` is a hash set of newly allocated datatype-reachable objects
- method root insertion: when new specializations generate new roots, these roots must be inserted into
  method root tables
- backedges & invalidation: external edges have to be checked against the running system and any invalidations executed.

Encoding of a pointer:
- in the location of the pointer, we initially write zero padding
- for both relocs_list and gctags_list, we write loc/backrefid (for gctags_list this is handled by the caller of write_gctaggedfield,
  for relocs_list it's handled by write_pointerfield)
- when writing to disk, both call get_reloc_for_item, and its return value (subject to modification by gc bits)
  ends up being written into the data stream (s->s), and the data stream's position written to s->relocs

External links:
- location holds the offset
- loc/0 in relocs_list

*/
#include <stdlib.h>
#include <string.h>
#include <stdio.h> // printf
#include <inttypes.h> // PRIxPTR

#include <zstd.h>

#include "julia.h"
#include "julia_internal.h"
#include "julia_gcext.h"
#include "builtin_proto.h"
#include "processor.h"
#include "serialize.h"

#ifdef _OS_WINDOWS_
#include <memoryapi.h>
#else
#include <dlfcn.h>
#include <sys/mman.h>
#endif

#include "valgrind.h"
#include "julia_assert.h"

static const size_t WORLD_AGE_REVALIDATION_SENTINEL = 0x1;
JL_DLLEXPORT size_t jl_require_world = ~(size_t)0;
JL_DLLEXPORT _Atomic(size_t) jl_first_image_replacement_world = ~(size_t)0;

// This structure is used to store hash tables for the memoization
// of queries in staticdata.c (currently only `type_in_worklist`).
typedef struct {
    htable_t type_in_worklist;
} jl_query_cache;

static void init_query_cache(jl_query_cache *cache) JL_NOTSAFEPOINT
{
    htable_new(&cache->type_in_worklist, 0);
}

static void destroy_query_cache(jl_query_cache *cache) JL_NOTSAFEPOINT
{
    htable_free(&cache->type_in_worklist);
}

#include "staticdata_utils.c"
#include "precompile_utils.c"

#ifdef __cplusplus
extern "C" {
#endif

// TODO: put WeakRefs on the weak_refs list during deserialization
// TODO: handle finalizers

#define NUM_TAGS    6

// An array of special references that need to be restored from the sysimg
static void get_tags(jl_value_t **tags[NUM_TAGS])
{
    // Make sure to keep an extra slot at the end to sentinel length
    unsigned int i = 0;
#define INSERT_TAG(sym) tags[i++] = (jl_value_t**)&(sym)
    INSERT_TAG(jl_method_table);
    INSERT_TAG(jl_module_init_order);
    INSERT_TAG(jl_typeinf_func);
    INSERT_TAG(jl_compile_and_emit_func);
    INSERT_TAG(jl_libdl_dlopen_func);
    // n.b. must update NUM_TAGS when you add something here
#undef INSERT_TAG
    assert(i == NUM_TAGS - 1);
    tags[i] = NULL;
}

// hash of definitions for predefined tagged object
static htable_t symbol_table;
static uintptr_t nsym_tag;
// array of definitions for the predefined tagged object types
// (reverse of symbol_table)
static arraylist_t deser_sym;

static htable_t serialization_order; // to break cycles, mark all objects that are serialized
static htable_t nullptrs;
// FIFO queue for objects to be serialized. Anything requiring fixup upon deserialization
// must be "toplevel" in this queue. For types, parameters and field types must appear
// before the "wrapper" type so they can be properly recached against the running system.
static arraylist_t serialization_queue;
static arraylist_t layout_table;     // cache of `position(s)` for each `id` in `serialization_order`
static arraylist_t object_worklist;  // used to mimic recursion by jl_serialize_reachable

// Permanent list of void* (begin, end+1) pairs of system/package images we've loaded previously
// together with their module build_ids (used for external linkage)
// jl_linkage_blobs.items[2i:2i+1] correspond to build_ids[i]   (0-offset indexing)
arraylist_t jl_linkage_blobs;
arraylist_t jl_image_relocs;
// Keep track of which image corresponds to which top module.
arraylist_t jl_top_mods;

// Eytzinger tree of images. Used for very fast jl_object_in_image queries
// See https://algorithmica.org/en/eytzinger
arraylist_t eytzinger_image_tree;
arraylist_t eytzinger_idxs;
static uintptr_t img_min;
static uintptr_t img_max;

// HT_NOTFOUND is a valid integer ID, so we store the integer ids mangled.
// This pair of functions mangles/demanges
static size_t from_seroder_entry(void *entry) JL_NOTSAFEPOINT
{
    return (size_t)((char*)entry - (char*)HT_NOTFOUND - 1);
}

static void *to_seroder_entry(size_t idx) JL_NOTSAFEPOINT
{
    return (void*)((char*)HT_NOTFOUND + 1 + idx);
}

static htable_t new_methtables;
//static size_t precompilation_world;

static int ptr_cmp(const void *l, const void *r) JL_NOTSAFEPOINT
{
    uintptr_t left = *(const uintptr_t*)l;
    uintptr_t right = *(const uintptr_t*)r;
    return (left > right) - (left < right);
}

// Build an eytzinger tree from a sorted array
static int eytzinger(uintptr_t *src, uintptr_t *dest, size_t i, size_t k, size_t n) JL_NOTSAFEPOINT
{
    if (k <= n) {
        i = eytzinger(src, dest, i, 2 * k, n);
        dest[k-1] = src[i];
        i++;
        i = eytzinger(src, dest, i, 2 * k + 1, n);
    }
    return i;
}

static size_t eyt_obj_idx(jl_value_t *obj) JL_NOTSAFEPOINT
{
    size_t n = eytzinger_image_tree.len - 1;
    if (n == 0)
        return n;
    assert(n % 2 == 0 && "Eytzinger tree not even length!");
    uintptr_t cmp = (uintptr_t) obj;
    if (cmp <= img_min || cmp > img_max)
        return n;
    uintptr_t *tree = (uintptr_t*)eytzinger_image_tree.items;
    size_t k = 1;
    // note that k preserves the history of how we got to the current node
    while (k <= n) {
        int greater = (cmp > tree[k - 1]);
        k <<= 1;
        k |= greater;
    }
    // Free to assume k is nonzero, since we start with k = 1
    // and cmp > gc_img_min
    // This shift does a fast revert of the path until we get
    // to a node that evaluated less than cmp.
    k >>= (__builtin_ctzll(k) + 1);
    assert(k != 0);
    assert(k <= n && "Eytzinger tree index out of bounds!");
    assert(tree[k - 1] < cmp && "Failed to find lower bound for object!");
    return k - 1;
}

//used in staticdata.c after we add an image
void rebuild_image_blob_tree(void) JL_NOTSAFEPOINT
{
    size_t inc = 1 + jl_linkage_blobs.len - eytzinger_image_tree.len;
    assert(eytzinger_idxs.len == eytzinger_image_tree.len);
    assert(eytzinger_idxs.max == eytzinger_image_tree.max);
    arraylist_grow(&eytzinger_idxs, inc);
    arraylist_grow(&eytzinger_image_tree, inc);
    eytzinger_idxs.items[eytzinger_idxs.len - 1] = (void*)jl_linkage_blobs.len;
    eytzinger_image_tree.items[eytzinger_image_tree.len - 1] = (void*)1; // outside image
    for (size_t i = 0; i < jl_linkage_blobs.len; i++) {
        assert((uintptr_t) jl_linkage_blobs.items[i] % 4 == 0 && "Linkage blob not 4-byte aligned!");
        // We abuse the pointer here a little so that a couple of properties are true:
        // 1. a start and an end are never the same value. This simplifies the binary search.
        // 2. ends are always after starts. This also simplifies the binary search.
        // We assume that there exist no 0-size blobs, but that's a safe assumption
        // since it means nothing could be there anyways
        uintptr_t val = (uintptr_t) jl_linkage_blobs.items[i];
        eytzinger_idxs.items[i] = (void*)(val + (i & 1));
    }
    qsort(eytzinger_idxs.items, eytzinger_idxs.len - 1, sizeof(void*), ptr_cmp);
    img_min = (uintptr_t) eytzinger_idxs.items[0];
    img_max = (uintptr_t) eytzinger_idxs.items[eytzinger_idxs.len - 2] + 1;
    eytzinger((uintptr_t*)eytzinger_idxs.items, (uintptr_t*)eytzinger_image_tree.items, 0, 1, eytzinger_idxs.len - 1);
    // Reuse the scratch memory to store the indices
    // Still O(nlogn) because binary search
    for (size_t i = 0; i < jl_linkage_blobs.len; i ++) {
        uintptr_t val = (uintptr_t) jl_linkage_blobs.items[i];
        // This is the same computation as in the prior for loop
        uintptr_t eyt_val = val + (i & 1);
        size_t eyt_idx = eyt_obj_idx((jl_value_t*)(eyt_val + 1)); assert(eyt_idx < eytzinger_idxs.len - 1);
        assert(eytzinger_image_tree.items[eyt_idx] == (void*)eyt_val && "Eytzinger tree failed to find object!");
        if (i & 1)
            eytzinger_idxs.items[eyt_idx] = (void*)n_linkage_blobs();
        else
            eytzinger_idxs.items[eyt_idx] = (void*)(i / 2);
    }
}

static int eyt_obj_in_img(jl_value_t *obj) JL_NOTSAFEPOINT
{
    assert((uintptr_t) obj % 4 == 0 && "Object not 4-byte aligned!");
    int idx = eyt_obj_idx(obj);
    // Now we use a tiny trick: tree[idx] & 1 is whether or not tree[idx] is a
    // start (0) or an end (1) of a blob. If it's a start, then the object is
    // in the image, otherwise it is not.
    int in_image = ((uintptr_t)eytzinger_image_tree.items[idx] & 1) == 0;
    return in_image;
}

size_t external_blob_index(jl_value_t *v) JL_NOTSAFEPOINT
{
    assert((uintptr_t) v % 4 == 0 && "Object not 4-byte aligned!");
    int eyt_idx = eyt_obj_idx(v);
    // We fill the invalid slots with the length, so we can just return that
    size_t idx = (size_t) eytzinger_idxs.items[eyt_idx];
    return idx;
}

JL_DLLEXPORT uint8_t jl_object_in_image(jl_value_t *obj) JL_NOTSAFEPOINT
{
    return eyt_obj_in_img(obj);
}

// Map an object to it's "owning" top module
JL_DLLEXPORT jl_value_t *jl_object_top_module(jl_value_t* v) JL_NOTSAFEPOINT
{
    size_t idx = external_blob_index(v);
    size_t lbids = n_linkage_blobs();
    if (idx < lbids) {
        return (jl_value_t*)jl_top_mods.items[idx];
    }
    // The object is runtime allocated
    return (jl_value_t*)jl_nothing;
}

// hash of definitions for predefined function pointers
// (reverse is jl_builtin_f_addrs)
static htable_t fptr_to_id;

void *native_functions;   // opaque jl_native_code_desc_t blob used for fetching data from LLVM

// table of struct field addresses to rewrite during saving
static htable_t field_replace;
static htable_t bits_replace;


typedef struct {
    ios_t *s;                   // the main stream
    ios_t *const_data;          // GC-invisible internal data (e.g., datatype layouts, list-like typename fields, foreign types, internal arrays)
    ios_t *symbols;             // names (char*) of symbols (some may be referenced by pointer in generated code)
    ios_t *relocs;              // for (de)serializing relocs_list and gctags_list
    ios_t *gvar_record;         // serialized array mapping gvid => spos
    ios_t *fptr_record;         // serialized array mapping fptrid => spos
    arraylist_t memowner_list;  // a list of memory locations that have shared owners
    arraylist_t memref_list;    // a list of memoryref locations
    arraylist_t relocs_list;    // a list of (location, target) pairs, see description at top
    arraylist_t gctags_list;    //      "
    arraylist_t uniquing_types; // a list of locations that reference types that must be de-duplicated
    arraylist_t uniquing_super; // a list of datatypes, used in super fields, that need to be marked in uniquing_types once they are reached, for handling unique-ing of them on deserialization
    arraylist_t uniquing_objs;  // a list of locations that reference non-types that must be de-duplicated
    arraylist_t fixup_types;    // a list of locations of types requiring (re)caching
    arraylist_t fixup_objs;     // a list of locations of objects requiring (re)caching
    // mapping from a buildid_idx to a depmods_idx
    jl_array_t *buildid_depmods_idxs;
    // Re-linking against a rebuilt dependency (JULIA_PKGIMAGE_RELINK). `relink_map` is the
    // image's import table sorted by the (deps-index, offset) pair that every external
    // reference already carries, with the object each entry resolved to; `relink_deps`
    // marks the dependencies whose blob moved, which are the only ones that need it.
    // Sorted-plus-binary-search rather than a hash table: which entry owns a reference is
    // then settled by construction, at no cost to the references that do not need it.
    struct jl_relink_ent_t *relink_map;
    size_t relink_nmap;
    const uint8_t *relink_deps;
    size_t relink_ndeps;
    // record of build_ids for all external linkages, in order of serialization for the current sysimg/pkgimg
    // conceptually, the base pointer for the jth externally-linked item is determined from
    //     i = findfirst(==(link_ids[j]), build_ids)
    //     blob_base = jl_linkage_blobs.items[2i]                     # 0-offset indexing
    // We need separate lists since they are intermingled at creation but split when written.
    jl_array_t *link_ids_relocs;
    jl_array_t *link_ids_gctags;
    jl_array_t *link_ids_gvars;
    jl_array_t *link_ids_external_fnvars;
    jl_array_t *method_roots_list;
    htable_t method_roots_index;
    // Import table: the distinct objects this image references in *other* images, in
    // first-reference order. `import_objs[i]` lives in the image whose deps-index is
    // `import_deps.items[i]`. Today this is descriptive only; it is the table a
    // content-keyed external-reference scheme would serialize, so that a rebuilt
    // dependency can be re-resolved by key instead of invalidating this image.
    arraylist_t import_objs;
    arraylist_t import_deps;
    htable_t import_index;      // jl_value_t* -> 1-based index into import_objs
    uint64_t worklist_key;
    jl_query_cache *query_cache;
    jl_ptls_t ptls;
    jl_image_t *image;
    int8_t incremental;
} jl_serializer_state;

static jl_value_t *jl_bigint_type = NULL;
static jl_debuginfo_t *jl_nulldebuginfo;
static int gmp_limb_size = 0;

#ifdef _P64
#define RELOC_TAG_OFFSET 61
#define DEPS_IDX_OFFSET 40    // only on 64-bit can we encode the dependency-index as part of the tagged reloc
#else
// this supports up to 8 RefTags, 512MB of pointer data, and 4/2 (64/32-bit) GB of constant data.
#define RELOC_TAG_OFFSET 29
#define DEPS_IDX_OFFSET RELOC_TAG_OFFSET
#endif


// Tags of category `t` are located at offsets `t << RELOC_TAG_OFFSET`
// Consequently there is room for 2^RELOC_TAG_OFFSET pointers, etc
enum RefTags {
    DataRef,            // mutable data
    ConstDataRef,       // constant data (e.g., layouts)
    TagRef,             // items serialized via their tags
    SymbolRef,          // symbols
    FunctionRef,        // functions
    SysimageLinkage,    // reference to the sysimage (from pkgimage)
    ExternalLinkage     // reference to some other pkgimage
};

#define SYS_EXTERNAL_LINK_UNIT sizeof(void*)

// calling conventions for internal entry points.
// this is used to set the method-instance->invoke field
typedef enum {
    JL_API_NULL,
    JL_API_BOXED,
    JL_API_CONST,
    JL_API_WITH_PARAMETERS,
    JL_API_OC_CALL,
    JL_API_INTERPRETED,
    JL_API_BUILTIN,
    JL_API_MAX
} jl_callingconv_t;

// Sub-divisions of some RefTags
const uintptr_t BuiltinFunctionTag = ((uintptr_t)1 << (RELOC_TAG_OFFSET - 1));


#if RELOC_TAG_OFFSET <= 32
typedef uint32_t reloc_t;
#else
typedef uint64_t reloc_t;
#endif
static void write_reloc_t(ios_t *s, uintptr_t reloc_id) JL_NOTSAFEPOINT
{
    if (sizeof(reloc_t) <= sizeof(uint32_t)) {
        assert(reloc_id < UINT32_MAX);
        write_uint32(s, reloc_id);
    }
    else {
        write_uint64(s, reloc_id);
    }
}

// Reporting to PkgCacheInspector
typedef struct {
    size_t sysdata;
    size_t isbitsdata;
    size_t symboldata;
    size_t tagslist;
    size_t reloclist;
    size_t gvarlist;
    size_t fptrlist;
} pkgcachesizes;

// --- Static Compile ---
static jl_image_buf_t jl_sysimage_buf = { JL_IMAGE_KIND_NONE };

static inline uintptr_t *sysimg_gvars(const char *base, const int32_t *offsets, size_t idx)
{
    return (uintptr_t*)(base + offsets[idx]);
}

JL_DLLEXPORT int jl_running_on_valgrind(void)
{
    return RUNNING_ON_VALGRIND;
}

// --- serializer ---

#define NBOX_C 1024

// DebugInfo owned by another image is copied into the image being written instead of
// being referenced across images: it is immutable content with no stable name, so a
// cross-image reference could only be keyed by a content digest, which a loader cannot
// invert to locate the object again. Copying trades image size for keeping every
// import-table entry resolvable. The `edges` simplevectors holding nested DebugInfo are
// reached by the same path and copied for the same reason (their key would be a list of
// those same digests).
static int jl_copy_instead_of_import(jl_value_t *v) JL_NOTSAFEPOINT
{
    if (jl_typetagis(v, jl_debuginfo_type))
        return 1;
    if (jl_is_svec(v)) {
        size_t i, l = jl_svec_len(v);
        if (l == 0)
            return 0;
        for (i = 0; i < l; i++) {
            jl_value_t *e = jl_svecref(v, i);
            if (e == NULL || !jl_typetagis(e, jl_debuginfo_type))
                return 0;
        }
        return 1;
    }
    return 0;
}

static int jl_needs_serialization(jl_serializer_state *s, jl_value_t *v) JL_NOTSAFEPOINT
{
    // ignore items that are given a special relocation representation
    if (s->incremental && jl_object_in_image(v) && !jl_copy_instead_of_import(v))
        return 0;

    if (v == NULL || jl_is_symbol(v) || v == jl_nothing) {
        return 0;
    }
    else if (jl_typetagis(v, jl_int64_tag << 4)) {
        int64_t i64 = *(int64_t*)v + NBOX_C / 2;
        if ((uint64_t)i64 < NBOX_C)
            return 0;
    }
    else if (jl_typetagis(v, jl_int32_tag << 4)) {
        int32_t i32 = *(int32_t*)v + NBOX_C / 2;
        if ((uint32_t)i32 < NBOX_C)
            return 0;
    }
    else if (jl_typetagis(v, jl_uint8_tag << 4)) {
        return 0;
    }
    else if (v == (jl_value_t*)s->ptls->root_task) {
        return 0;
    }

    return 1;
}

static int caching_tag(jl_value_t *v, jl_query_cache *query_cache) JL_NOTSAFEPOINT
{
    if (jl_is_method_instance(v)) {
        jl_method_instance_t *mi = (jl_method_instance_t*)v;
        jl_value_t *m = mi->def.value;
        if (jl_is_method(m) && jl_object_in_image(m))
            return 1 + type_in_worklist(mi->specTypes, query_cache);
    }
    if (jl_is_binding(v)) {
        jl_globalref_t *gr = ((jl_binding_t*)v)->globalref;
        if (!gr)
            return 0;
        if (!jl_object_in_image((jl_value_t*)gr->mod))
            return 0;
        return 1;
    }
    if (jl_is_datatype(v)) {
        jl_datatype_t *dt = (jl_datatype_t*)v;
        if (jl_is_tuple_type(dt) ? !dt->isconcretetype : dt->hasfreetypevars)
            return 0; // aka !is_cacheable from jltypes.c
        if (jl_object_in_image((jl_value_t*)dt->name))
            return 1 + type_in_worklist(v, query_cache);
    }
    jl_value_t *dtv = jl_typeof(v);
    if (jl_is_datatype_singleton((jl_datatype_t*)dtv)) {
        return 1 - type_in_worklist(dtv, query_cache); // these are already recached in the datatype in the image
    }
    return 0;
}

static int needs_recaching(jl_value_t *v, jl_query_cache *query_cache) JL_NOTSAFEPOINT
{
    return caching_tag(v, query_cache) == 2;
}

static int needs_uniquing(jl_value_t *v, jl_query_cache *query_cache) JL_NOTSAFEPOINT
{
    assert(!jl_object_in_image(v) || jl_copy_instead_of_import(v));
    return caching_tag(v, query_cache) == 1;
}

static void record_field_change(jl_value_t **addr, jl_value_t *newval) JL_NOTSAFEPOINT
{
    if (*addr != newval)
        ptrhash_put(&field_replace, (void*)addr, newval);
}

static jl_value_t *get_replaceable_field(jl_value_t **addr, int mutabl) JL_GC_DISABLED
{
    jl_value_t *fld = (jl_value_t*)ptrhash_get(&field_replace, addr);
    if (fld == HT_NOTFOUND) {
        fld = *addr;
        if (mutabl && fld && jl_is_cpointer_type(jl_typeof(fld)) && jl_unbox_voidpointer(fld) != NULL && jl_unbox_voidpointer(fld) != (void*)(uintptr_t)-1) {
            void **nullval = ptrhash_bp(&nullptrs, (void*)jl_typeof(fld));
            if (*nullval == HT_NOTFOUND) {
                void *C_NULL = NULL;
                *nullval = (void*)jl_new_bits(jl_typeof(fld), &C_NULL);
            }
            fld = (jl_value_t*)*nullval;
        }
        return fld;
    }
    return fld;
}

static uintptr_t jl_fptr_id(void *fptr)
{
    void **pbp = ptrhash_bp(&fptr_to_id, fptr);
    if (*pbp == HT_NOTFOUND || fptr == NULL)
        return 0;
    else
        return *(uintptr_t*)pbp;
}

static int effects_foldable(uint32_t effects)
{
    // N.B.: This needs to be kept in sync with Core.Compiler.is_foldable(effects, true)
    return ((effects & 0x7) == 0) && // is_consistent(effects)
           (((effects >> 10) & 0x03) == 0) && // is_noub(effects)
           (((effects >> 3) & 0x03) == 0) && // is_effect_free(effects)
           ((effects >> 6) & 0x01); // is_terminates(effects)
}


// `jl_queue_for_serialization` adds items to `serialization_order`
#define jl_queue_for_serialization(s, v) jl_queue_for_serialization_((s), (jl_value_t*)(v), 1, 0)
static void jl_queue_for_serialization_(jl_serializer_state *s, jl_value_t *v, int recursive, int immediate) JL_GC_DISABLED;

static void jl_queue_module_for_serialization(jl_serializer_state *s, jl_module_t *m) JL_GC_DISABLED
{
    jl_queue_for_serialization(s, m->name);
    jl_queue_for_serialization(s, m->parent);
    if (!jl_options.strip_metadata)
        jl_queue_for_serialization(s, m->file);
    jl_queue_for_serialization(s, jl_atomic_load_relaxed(&m->bindingkeyset));
    if (jl_options.trim) {
        jl_queue_for_serialization_(s, (jl_value_t*)jl_atomic_load_relaxed(&m->bindings), 0, 1);
        jl_svec_t *table = jl_atomic_load_relaxed(&m->bindings);
        for (size_t i = 0; i < jl_svec_len(table); i++) {
            jl_binding_t *b = (jl_binding_t*)jl_svecref(table, i);
            if ((void*)b == jl_nothing)
                break;
            jl_value_t *val = jl_get_binding_value_in_world(b, jl_atomic_load_relaxed(&jl_world_counter));
            // keep binding objects that are defined in the latest world and ...
            if (val &&
                // ... point to modules ...
                (jl_is_module(val) ||
                 // ... or point to __init__ methods ...
                 !strcmp(jl_symbol_name(b->globalref->name), "__init__") ||
                 // ... or point to Base functions accessed by the runtime
                 (m == jl_base_module && (!strcmp(jl_symbol_name(b->globalref->name), "wait") ||
                                          !strcmp(jl_symbol_name(b->globalref->name), "task_done_hook") ||
                                          !strcmp(jl_symbol_name(b->globalref->name), "_uv_hook_close"))))) {
                jl_queue_for_serialization(s, b);
            }
        }
    }
    else {
        jl_queue_for_serialization(s, jl_atomic_load_relaxed(&m->bindings));
    }

    for (size_t i = 0; i < module_usings_length(m); i++) {
        jl_queue_for_serialization(s, module_usings_getmod(m, i));
    }

    if (jl_options.trim || jl_options.strip_ir) {
        record_field_change((jl_value_t**)&m->usings_backedges, jl_nothing);
        record_field_change((jl_value_t**)&m->scanned_methods, jl_nothing);
    }
    else {
        jl_queue_for_serialization(s, m->usings_backedges);
        jl_queue_for_serialization(s, m->scanned_methods);
    }
}

static int codeinst_may_be_runnable(jl_code_instance_t *ci, int incremental) {
    size_t max_world = jl_atomic_load_relaxed(&ci->max_world);
    if (max_world == ~(size_t)0)
        return 1;
    if (incremental)
        return 0;
    return jl_atomic_load_relaxed(&ci->min_world) <= jl_typeinf_world && jl_typeinf_world <= max_world;
}

// Anything that requires uniquing or fixing during deserialization needs to be "toplevel"
// in serialization (i.e., have its own entry in `serialization_order`). Consequently,
// objects that act as containers for other potentially-"problematic" objects must add such "children"
// to the queue.
// Most objects use preorder traversal. But things that need uniquing require postorder:
// you want to handle uniquing of `Dict{String,Float64}` before you tackle `Vector{Dict{String,Float64}}`.
// Uniquing is done in `serialization_order`, so the very first mention of such an object must
// be the "source" rather than merely a cross-reference.
static void jl_insert_into_serialization_queue(jl_serializer_state *s, jl_value_t *v, int recursive, int immediate) JL_GC_DISABLED
{
    jl_datatype_t *t = (jl_datatype_t*)jl_typeof(v);
    jl_queue_for_serialization_(s, (jl_value_t*)t, 1, immediate);
    const jl_datatype_layout_t *layout = t->layout;

    if (!recursive)
        goto done_fields;

    if (s->incremental && jl_is_datatype(v) && immediate) {
        jl_datatype_t *dt = (jl_datatype_t*)v;
        // ensure all type parameters are recached
        jl_queue_for_serialization_(s, (jl_value_t*)dt->parameters, 1, 1);
        if (jl_is_datatype_singleton(dt) && needs_uniquing(dt->instance, s->query_cache)) {
            assert(jl_needs_serialization(s, dt->instance)); // should be true, since we visited dt
            // do not visit dt->instance for our template object as it leads to unwanted cycles here
            // (it may get serialized from elsewhere though)
            record_field_change(&dt->instance, jl_nothing);
        }
        goto done_fields; // for now
    }
    if (jl_is_method_instance(v)) {
        jl_method_instance_t *mi = (jl_method_instance_t*)v;
        if (s->incremental) {
            jl_value_t *def = mi->def.value;
            if (needs_uniquing(v, s->query_cache)) {
                // we only need 3 specific fields of this (the rest are not used)
                jl_queue_for_serialization(s, mi->def.value);
                jl_queue_for_serialization(s, mi->specTypes);
                jl_queue_for_serialization(s, (jl_value_t*)mi->sparam_vals);
                goto done_fields;
            }
            else if (jl_is_method(def) && jl_object_in_image(def)) {
                // we only need 3 specific fields of this (the rest are restored afterward, if valid)
                // in particular, cache is repopulated by jl_mi_cache_insert for all foreign function,
                // so must not be present here
                record_field_change((jl_value_t**)&mi->cache, NULL);
            }
            else {
                assert(!needs_recaching(v, s->query_cache));
            }
            // Any back-edges will be re-validated and added by staticdata.jl, so
            // drop them from the image here
            record_field_change((jl_value_t**)&mi->backedges, NULL);
            // n.b. opaque closures cannot be inspected and relied upon like a
            // normal method since they can get improperly introduced by generated
            // functions, so if they appeared at all, we will probably serialize
            // them wrong and segfault. The jl_code_for_staged function should
            // prevent this from happening, so we do not need to detect that user
            // error now.
        }
        // don't recurse into all backedges memory (yet)
        jl_value_t *backedges = get_replaceable_field((jl_value_t**)&mi->backedges, 1);
        if (backedges) {
            assert(!jl_options.trim && !jl_options.strip_ir);
            jl_queue_for_serialization_(s, (jl_value_t*)((jl_array_t*)backedges)->ref.mem, 0, 1);
            size_t i = 0, n = jl_array_nrows(backedges);
            while (i < n) {
                jl_value_t *invokeTypes;
                jl_code_instance_t *caller;
                i = get_next_edge((jl_array_t*)backedges, i, &invokeTypes, &caller);
                if (invokeTypes)
                    jl_queue_for_serialization(s, invokeTypes);
            }
        }
    }
    if (jl_is_binding(v)) {
        jl_binding_t *b = (jl_binding_t*)v;
        if (s->incremental && needs_uniquing(v, s->query_cache)) {
            jl_queue_for_serialization(s, b->globalref->mod);
            jl_queue_for_serialization(s, b->globalref->name);
            goto done_fields;
        }
        if (jl_options.trim || jl_options.strip_ir) {
            record_field_change((jl_value_t**)&b->backedges, NULL);
        }
        else {
            // don't recurse into all backedges memory (yet)
            jl_value_t *backedges = get_replaceable_field((jl_value_t**)&b->backedges, 1);
            if (backedges) {
                jl_queue_for_serialization_(s, (jl_value_t*)((jl_array_t*)backedges)->ref.mem, 0, 1);
                for (size_t i = 0, n = jl_array_nrows(backedges); i < n; i++) {
                    jl_value_t *b = jl_array_ptr_ref(backedges, i);
                    if (!jl_is_code_instance(b) && !jl_is_method_instance(b) && !jl_is_method(b)) // otherwise usually a Binding?
                        jl_queue_for_serialization(s, b);
                }
            }
        }
    }
    if (s->incremental && jl_is_globalref(v)) {
        jl_globalref_t *gr = (jl_globalref_t*)v;
        if (jl_object_in_image((jl_value_t*)gr->mod)) {
            record_field_change((jl_value_t**)&gr->binding, NULL);
        }
    }
    if (jl_is_typename(v)) {
        jl_typename_t *tn = (jl_typename_t*)v;
        // don't recurse into several fields (yet)
        jl_queue_for_serialization_(s, (jl_value_t*)jl_atomic_load_relaxed(&tn->cache), 0, 1);
        jl_queue_for_serialization_(s, (jl_value_t*)jl_atomic_load_relaxed(&tn->linearcache), 0, 1);
        if (s->incremental) {
            assert(!jl_object_in_image((jl_value_t*)tn->module));
            assert(!jl_object_in_image((jl_value_t*)tn->wrapper));
        }
    }
    if (jl_is_mtable(v)) {
        jl_methtable_t *mt = (jl_methtable_t*)v;
        // Any back-edges will be re-validated and added by staticdata.jl, so
        // drop them from the image here
        if (s->incremental || jl_options.trim || jl_options.strip_ir) {
            record_field_change((jl_value_t**)&mt->backedges, jl_an_empty_memory_any);
        }
        else {
            // don't recurse into all backedges memory (yet)
            jl_value_t *allbackedges = get_replaceable_field((jl_value_t**)&mt->backedges, 1);
            jl_queue_for_serialization_(s, allbackedges, 0, 1);
            for (size_t i = 0, n = ((jl_genericmemory_t*)allbackedges)->length; i < n; i += 2) {
                jl_value_t *tn = jl_genericmemory_ptr_ref(allbackedges, i);
                jl_queue_for_serialization(s, tn);
                jl_value_t *backedges = jl_genericmemory_ptr_ref(allbackedges, i + 1);
                if (backedges && backedges != jl_nothing) {
                    jl_queue_for_serialization_(s, (jl_value_t*)((jl_array_t*)backedges)->ref.mem, 0, 1);
                    jl_queue_for_serialization(s, backedges);
                    for (size_t i = 0, n = jl_array_nrows(backedges); i < n; i += 2) {
                        jl_value_t *t = jl_array_ptr_ref(backedges, i);
                        assert(!jl_is_code_instance(t));
                        jl_queue_for_serialization(s, t);
                    }
                }
            }
        }
    }
    if (jl_is_code_instance(v)) {
        jl_code_instance_t *ci = (jl_code_instance_t*)v;
        jl_method_instance_t *mi = jl_get_ci_mi(ci);
        if (s->incremental) {
            // make sure we don't serialize other reachable cache entries of foreign methods
            // Should this now be:
            // if (ci !in ci->defs->cache)
            //     record_field_change((jl_value_t**)&ci->next, NULL);
            // Why are we checking that the method/module this originates from is in_image?
            // and then disconnect this CI?
            if (jl_object_in_image((jl_value_t*)mi->def.value)) {
                // TODO: if (ci in ci->defs->cache)
                record_field_change((jl_value_t**)&ci->next, NULL);
            }
        }
        jl_value_t *inferred = jl_atomic_load_relaxed(&ci->inferred);
        if (inferred && inferred != jl_nothing && !jl_is_uint8(inferred)) { // disregard if there is nothing here to delete (e.g. builtins, unspecialized)
            jl_method_t *def = mi->def.method;
            if (jl_is_method(def)) { // don't delete toplevel code
                int is_relocatable = !s->incremental || jl_is_code_info(inferred) ||
                    (jl_is_string(inferred) && jl_string_len(inferred) > 0 && jl_string_data(inferred)[jl_string_len(inferred) - 1]);
                int discard = 0;
                if (!is_relocatable) {
                    discard = 1;
                }
                else if (def->source == NULL) {
                    // don't delete code from optimized opaque closures that can't be reconstructed (and builtins)
                }
                else if (!codeinst_may_be_runnable(ci, s->incremental) || // delete all code that cannot run
                         jl_atomic_load_relaxed(&ci->invoke) == jl_fptr_const_return) { // delete all code that just returns a constant
                    discard = 1;
                }
                else if (native_functions && // don't delete any code if making a ji file
                         (ci->owner == jl_nothing) && // don't delete code for external interpreters
                         !effects_foldable(jl_atomic_load_relaxed(&ci->ipo_purity_bits)) && // don't delete code we may want for irinterp
                         jl_ir_inlining_cost(inferred) == UINT16_MAX) { // don't delete inlineable code
                    // delete the code now: if we thought it was worth keeping, it would have been converted to object code
                    discard = 1;
                }
                if (discard) {
                    // keep only the inlining cost, so inference can later decide if it is worth getting the source back
                    if (jl_is_string(inferred) || jl_is_code_info(inferred))
                        inferred = jl_box_uint8(jl_encode_inlining_cost(jl_ir_inlining_cost(inferred)));
                    else
                        inferred = jl_nothing;
                    record_field_change((jl_value_t**)&ci->inferred, inferred);
                }
                else if (s->incremental && jl_is_string(inferred)) {
                    // New roots for external methods
                    if (jl_object_in_image((jl_value_t*)def)) {
                        void **pfound = ptrhash_bp(&s->method_roots_index, def);
                        if (*pfound == HT_NOTFOUND) {
                            *pfound = def;
                            size_t nwithkey = nroots_with_key(def, s->worklist_key);
                            if (nwithkey) {
                                jl_array_ptr_1d_push(s->method_roots_list, (jl_value_t*)def);
                                jl_array_t *newroots = jl_alloc_vec_any(nwithkey);
                                jl_array_ptr_1d_push(s->method_roots_list, (jl_value_t*)newroots);
                                rle_iter_state rootiter = rle_iter_init(0);
                                uint64_t *rletable = NULL;
                                size_t nblocks2 = 0;
                                size_t nroots = jl_array_nrows(def->roots);
                                size_t k = 0;
                                if (def->root_blocks) {
                                    rletable = jl_array_data(def->root_blocks, uint64_t);
                                    nblocks2 = jl_array_nrows(def->root_blocks);
                                }
                                while (rle_iter_increment(&rootiter, nroots, rletable, nblocks2)) {
                                    if (rootiter.key == s->worklist_key) {
                                        jl_value_t *newroot = jl_array_ptr_ref(def->roots, rootiter.i);
                                        jl_queue_for_serialization(s, newroot);
                                        jl_array_ptr_set(newroots, k++, newroot);
                                    }
                                }
                                assert(k == nwithkey);
                            }
                        }
                    }
                }
            }
        }
    }

    if (immediate) // must be things that can be recursively handled, and valid as type parameters
        assert(jl_is_immutable(t) || jl_is_typevar(v) || jl_is_symbol(v) || jl_is_svec(v));

    if (layout->npointers == 0) {
        // bitstypes do not require recursion
    }
    else if (jl_is_svec(v)) {
        size_t i, l = jl_svec_len(v);
        jl_value_t **data = jl_svec_data(v);
        for (i = 0; i < l; i++) {
            jl_queue_for_serialization_(s, data[i], 1, immediate);
        }
    }
    else if (jl_is_array(v)) {
        jl_array_t *ar = (jl_array_t*)v;
        jl_value_t *mem = get_replaceable_field((jl_value_t**)&ar->ref.mem, 1);
        jl_queue_for_serialization_(s, mem, 1, immediate);
    }
    else if (jl_is_genericmemory(v)) {
        jl_genericmemory_t *m = (jl_genericmemory_t*)v;
        const char *data = (const char*)m->ptr;
        if (jl_genericmemory_how(m) == JL_GENERICMEMORY_STRINGOWNED) {
            assert(jl_is_string(jl_genericmemory_data_owner_field(m)));
        }
        else if (layout->flags.arrayelem_isboxed) {
            size_t i, l = m->length;
            for (i = 0; i < l; i++) {
                jl_value_t *fld = get_replaceable_field(&((jl_value_t**)data)[i], 1);
                jl_queue_for_serialization_(s, fld, 1, immediate);
            }
        }
        else if (layout->first_ptr >= 0) {
            uint16_t elsz = layout->size;
            size_t i, l = m->length;
            size_t j, np = layout->npointers;
            for (i = 0; i < l; i++) {
                for (j = 0; j < np; j++) {
                    uint32_t ptr = jl_ptr_offset(t, j);
                    jl_value_t *fld = get_replaceable_field(&((jl_value_t**)data)[ptr], 1);
                    jl_queue_for_serialization_(s, fld, 1, immediate);
                }
                data += elsz;
            }
        }
    }
    else if (jl_is_module(v)) {
        jl_queue_module_for_serialization(s, (jl_module_t*)v);
    }
    else if (layout->nfields > 0) {
        if (jl_options.trim) {
            if (jl_is_method(v)) {
                jl_method_t *m = (jl_method_t *)v;
                if (jl_is_svec(jl_atomic_load_relaxed(&m->specializations)))
                    jl_queue_for_serialization_(s, (jl_value_t*)jl_atomic_load_relaxed(&m->specializations), 0, 1);
            }
            else if (jl_is_mtable(v)) {
                jl_methtable_t *mt = (jl_methtable_t*)v;
                jl_methtable_t *newmt = (jl_methtable_t*)ptrhash_get(&new_methtables, mt);
                if (newmt != HT_NOTFOUND)
                    record_field_change((jl_value_t **)&mt->defs, (jl_value_t*)jl_atomic_load_relaxed(&newmt->defs));
                else
                    record_field_change((jl_value_t **)&mt->defs, jl_nothing);
            }
            else if (jl_is_mcache(v)) {
                jl_methcache_t *mc = (jl_methcache_t*)v;
                jl_value_t *cache = jl_atomic_load_relaxed(&mc->cache);
                if (!jl_typetagis(cache, jl_typemap_entry_type) || ((jl_typemap_entry_t*)cache)->sig != jl_tuple_type) { // aka Builtins (maybe sometimes OpaqueClosure too)
                    record_field_change((jl_value_t **)&mc->cache, jl_nothing);
                }
                record_field_change((jl_value_t **)&mc->leafcache, jl_an_empty_memory_any);
            }
            // TODO: prune any partitions and partition data that has been deleted in the current world
            //else if (jl_is_binding(v)) {
            //    jl_binding_t *b = (jl_binding_t*)v;
            //}
            //else if (jl_is_binding_partition(v)) {
            //    jl_binding_partition_t *bpart = (jl_binding_partition_t*)v;
            //}
        }
        char *data = (char*)jl_data_ptr(v);
        size_t i, np = layout->npointers;
        size_t fldidx = 1;
        for (i = 0; i < np; i++) {
            uint32_t ptr = jl_ptr_offset(t, i);
            size_t offset = jl_ptr_offset(t, i) * sizeof(jl_value_t*);
            while (offset >= (fldidx == layout->nfields ? jl_datatype_size(t) : jl_field_offset(t, fldidx)))
                fldidx++;
            int mutabl = !jl_field_isconst(t, fldidx - 1);
            jl_value_t *fld = get_replaceable_field(&((jl_value_t**)data)[ptr], mutabl);
            jl_queue_for_serialization_(s, fld, 1, immediate);
        }
    }

done_fields: ;

    // We've encountered an item we need to cache
    void **bp = ptrhash_bp(&serialization_order, v);
    assert(*bp == (void*)(uintptr_t)-2);
    arraylist_push(&serialization_queue, (void*) v);
    size_t idx = serialization_queue.len - 1;
    assert(serialization_queue.len < ((uintptr_t)1 << RELOC_TAG_OFFSET) && "too many items to serialize");
    *bp = to_seroder_entry(idx);

    // DataType is very unusual, in that some of the fields need to be pre-order, and some
    // (notably super) must not be (even if `jl_queue_for_serialization_` would otherwise
    // try to promote itself to be immediate)
    if (s->incremental && jl_is_datatype(v) && immediate && recursive) {
        jl_datatype_t *dt = (jl_datatype_t*)v;
        void **bp = ptrhash_bp(&serialization_order, (void*)dt->super);
        if (*bp != (void*)-2) {
            // if super is already on the stack of things to handle when this returns, do
            // not try to handle it now
            jl_queue_for_serialization_(s, (jl_value_t*)dt->super, 1, immediate);
        }
        immediate = 0;
        char *data = (char*)jl_data_ptr(v);
        size_t i, np = layout->npointers;
        for (i = 0; i < np; i++) {
            uint32_t ptr = jl_ptr_offset(t, i);
            if (ptr * sizeof(jl_value_t*) == offsetof(jl_datatype_t, super))
                continue; // skip the super field, since it might not be quite validly ordered
            int mutabl = 1;
            jl_value_t *fld = get_replaceable_field(&((jl_value_t**)data)[ptr], mutabl);
            jl_queue_for_serialization_(s, fld, 1, immediate);
        }
    }
}


static void jl_queue_for_serialization_(jl_serializer_state *s, jl_value_t *v, int recursive, int immediate) JL_GC_DISABLED
{
    if (!jl_needs_serialization(s, v))
        return;

    jl_datatype_t *t = (jl_datatype_t*)jl_typeof(v);
    // check early from errors, so we have a little bit of contextual state for debugging them
    if (t == jl_task_type) {
        jl_error("Task cannot be serialized");
    }
    if (s->incremental && needs_uniquing(v, s->query_cache) && t == jl_binding_type) {
        jl_binding_t *b = (jl_binding_t*)v;
        if (b->globalref == NULL)
            jl_error("Binding cannot be serialized"); // no way (currently) to recover its identity
    }
    if (jl_is_foreign_type(t) == 1) {
        jl_error("Cannot serialize instances of foreign datatypes");
    }

    // Items that require postorder traversal must visit their children prior to insertion into
    // the worklist/serialization_order (and also before their first use)
    if (s->incremental && !immediate) {
        if (jl_is_datatype(t) && needs_uniquing(v, s->query_cache))
            immediate = 1;
        if (jl_is_datatype_singleton((jl_datatype_t*)t) && needs_uniquing(v, s->query_cache))
            immediate = 1;
    }

    void **bp = ptrhash_bp(&serialization_order, v);
    assert(!immediate || *bp != (void*)(uintptr_t)-2);
    if (*bp == HT_NOTFOUND)
        *bp = (void*)(uintptr_t)-1; // now enqueued
    else if (!s->incremental || !immediate || !recursive || *bp != (void*)(uintptr_t)-1)
        return;

    if (immediate) {
        *bp = (void*)(uintptr_t)-2; // now immediate
        jl_insert_into_serialization_queue(s, v, recursive, immediate);
    }
    else {
        arraylist_push(&object_worklist, (void*)v);
    }
}

// Do a pre-order traversal of the to-serialize worklist, in the identical order
// to the calls to jl_queue_for_serialization would occur in a purely recursive
// implementation, but without potentially running out of stack.
static void jl_serialize_reachable(jl_serializer_state *s) JL_GC_DISABLED
{
    size_t i, prevlen = 0;
    while (object_worklist.len) {
        // reverse!(object_worklist.items, prevlen:end);
        // prevlen is the index of the first new object
        for (i = prevlen; i < object_worklist.len; i++) {
            size_t j = object_worklist.len - i + prevlen - 1;
            void *tmp = object_worklist.items[i];
            object_worklist.items[i] = object_worklist.items[j];
            object_worklist.items[j] = tmp;
        }
        prevlen = --object_worklist.len;
        jl_value_t *v = (jl_value_t*)object_worklist.items[prevlen];
        void **bp = ptrhash_bp(&serialization_order, (void*)v);
        assert(*bp != HT_NOTFOUND && *bp != (void*)(uintptr_t)-2);
        if (*bp == (void*)(uintptr_t)-1) { // might have been eagerly handled for post-order while in the lazy pre-order queue
            *bp = (void*)(uintptr_t)-2;
            jl_insert_into_serialization_queue(s, v, 1, 0);
        }
        else {
            assert(s->incremental);
        }
    }
}

static void ios_ensureroom(ios_t *s, size_t newsize) JL_NOTSAFEPOINT
{
    size_t prevsize = s->size;
    if (prevsize < newsize) {
        ios_trunc(s, newsize);
        assert(s->size == newsize);
        memset(&s->buf[prevsize], 0, newsize - prevsize);
    }
}

static void write_padding(ios_t *s, size_t nb) JL_NOTSAFEPOINT
{
    static const char zeros[16] = {0};
    while (nb > 16) {
        ios_write(s, zeros, 16);
        nb -= 16;
    }
    if (nb != 0)
        ios_write(s, zeros, nb);
}

static void write_pointer(ios_t *s) JL_NOTSAFEPOINT
{
    assert((ios_pos(s) & (sizeof(void*) - 1)) == 0 && "stream misaligned for writing a word-sized value");
    write_uint(s, 0);
}

// Records the buildid holding `v` and returns the tagged offset within the corresponding image
static uintptr_t add_external_linkage(jl_serializer_state *s, jl_value_t *v, jl_array_t *link_ids) JL_GC_DISABLED
{
    size_t i = external_blob_index(v);
    if (i < n_linkage_blobs()) {
        // Record `v` in the import table the first time we reference it. Distinct objects
        // are vastly fewer than references to them (~87:1 for a large package), which is
        // what makes a per-image key table cheap.
        if (ptrhash_get(&s->import_index, v) == HT_NOTFOUND) {
            arraylist_push(&s->import_objs, v);
            arraylist_push(&s->import_deps, (void*)(uintptr_t)i);
            ptrhash_put(&s->import_index, v, (void*)(uintptr_t)s->import_objs.len);
        }
        // We found the sysimg/pkg that this item links against
        // Compute the relocation code
        size_t offset = (uintptr_t)v - (uintptr_t)jl_linkage_blobs.items[2*i];
        assert((offset % SYS_EXTERNAL_LINK_UNIT) == 0);
        offset /= SYS_EXTERNAL_LINK_UNIT;
        assert(n_linkage_blobs() == jl_array_nrows(s->buildid_depmods_idxs));
        size_t depsidx = jl_array_data(s->buildid_depmods_idxs, uint32_t)[i]; // map from build_id_idx -> deps_idx
        assert(depsidx < INT32_MAX);
        if (depsidx < ((uintptr_t)1 << (RELOC_TAG_OFFSET - DEPS_IDX_OFFSET)) && offset < ((uintptr_t)1 << DEPS_IDX_OFFSET))
            // if it fits in a SysimageLinkage type, use that representation
            return ((uintptr_t)SysimageLinkage << RELOC_TAG_OFFSET) + ((uintptr_t)depsidx << DEPS_IDX_OFFSET) + offset;
        // otherwise, we store the image key in `link_ids`
        assert(link_ids && jl_is_array(link_ids));
        jl_array_grow_end(link_ids, 1);
        uint32_t *link_id_data  = jl_array_data(link_ids, uint32_t);  // wait until after the `grow`
        link_id_data[jl_array_nrows(link_ids) - 1] = depsidx;
        assert(offset < ((uintptr_t)1 << RELOC_TAG_OFFSET) && "offset to external image too large");
        return ((uintptr_t)ExternalLinkage << RELOC_TAG_OFFSET) + offset;
    }
    return 0;
}

// --- Content keys for external references -----------------------------------------
//
// A key identifies an object *within its owning image* using only information that
// survives a rebuild of that image: names and structure, never addresses. This is what
// lets a dependent image be re-resolved against a rebuilt dependency rather than
// discarded. Keys need only be unique per owning image, not globally.
//
// `extkey_write` returns 0 for object kinds that have no stable key yet; callers must
// treat that as "cannot re-resolve", i.e. fall back to rebuilding the dependent.

#define EXTKEY_MAX_DEPTH 8
// Types nest deeper than values do -- a method signature is routinely a dozen levels of
// `Tuple`/`Union`/`UnionAll` -- and unlike a value graph a type graph is finite and
// acyclic, so type nesting gets its own, looser budget. The value budget still bounds any
// alternation between the two: every hop back out of a type into a boxed parameter spends
// one unit of `depth`.
#define EXTKEY_MAX_TYPEDEPTH 40

static int extkey_write(ios_t *k, jl_value_t *v, int depth) JL_NOTSAFEPOINT;

// Which field defeats a CodeInstance key. They are 44% of the entries the writer cannot
// key at all, and the field that stops them decides whether that is fixable.
enum { EK_CI_MI, EK_CI_OWNER, EK_CI_RETTYPE, EK_CI_EDGES, EK_CI_EXCTYPE, EK_CI_RETCONST,
       EK_CI_NREASON };
static const char *const extkey_ci_reason[EK_CI_NREASON] = {
    "methodinstance", "owner", "rettype", "edges", "exctype", "rettype_const" };
static size_t extkey_ci_fail[EK_CI_NREASON];

// ...and, for the one field that turns out to account for nearly all of them, which kind
// of constant it is.
#define EK_RC_MAX 16
static const char *extkey_rc_name[EK_RC_MAX];
static size_t extkey_rc_count[EK_RC_MAX];
static int extkey_rc_n;
static void extkey_note_retconst(jl_value_t *c) JL_NOTSAFEPOINT
{
    const char *nm = jl_typeof_str(c);
    int q;
    for (q = 0; q < extkey_rc_n; q++)
        if (strcmp(extkey_rc_name[q], nm) == 0)
            break;
    if (q == extkey_rc_n && extkey_rc_n < EK_RC_MAX) {
        extkey_rc_name[extkey_rc_n] = nm;
        extkey_rc_count[extkey_rc_n] = 0;
        extkey_rc_n++;
    }
    if (q < extkey_rc_n)
        extkey_rc_count[q]++;
    // the type is what says whether the value is unkeyable in itself or only because its
    // own type is; print a few
    static int nshown = 0;
    if (getenv("JULIA_IMPORT_KEYS") && nshown < 4) {
        nshown++;
        ios_t t;
        ios_mem(&t, 128);
        int ok = extkey_write(&t, (jl_value_t*)jl_typeof(c), 0);
        ios_putc('\0', &t);
        jl_safe_printf("IMPORTKEYS_RETCONST_EX %s type=%s\n", nm, ok ? t.buf : "(unkeyable)");
        ios_close(&t);
    }
}

// Reverse index from an object to a module binding that holds it, so objects with no
// content identity can still be named by where they live. Built once per serialization.
typedef struct { jl_module_t *mod; jl_sym_t *name; } extkey_path_t;
static htable_t extkey_paths;
static int extkey_paths_ready = 0;

static const extkey_path_t *extkey_lookup_path(jl_value_t *v) JL_NOTSAFEPOINT
{
    if (!extkey_paths_ready)
        return NULL;
    void *p = ptrhash_get(&extkey_paths, v);
    return p == HT_NOTFOUND ? NULL : (const extkey_path_t*)p;
}

// Scan every loaded module's bindings, recording the first binding that holds each value.
// "First" is made deterministic by preferring the lexicographically smaller module path
// and then name, so two builds agree even if the tables are ordered differently.
static void extkey_build_paths(jl_array_t *mod_array) JL_NOTSAFEPOINT
{
    if (extkey_paths_ready || mod_array == NULL)
        return;
    htable_new(&extkey_paths, 0);
    extkey_paths_ready = 1;
    for (size_t mi = 0; mi < jl_array_nrows(mod_array); mi++) {
        jl_module_t *m = (jl_module_t*)jl_array_ptr_ref(mod_array, mi);
        if (!jl_is_module(m))
            continue;
        // (the module set scopes the table: see `extkey_reset_paths`)
        jl_svec_t *table = jl_atomic_load_relaxed(&m->bindings);
        for (size_t i = 0; i < jl_svec_len(table); i++) {
            jl_binding_t *b = (jl_binding_t*)jl_svecref(table, i);
            if ((void*)b == jl_nothing)
                break;
            if (!jl_is_binding(b) || b->globalref == NULL)
                continue;
            // Only constants qualify: a rebindable global could hold something else next
            // time, so its name would not be a stable identity for this value. The
            // `_debug_only` reader is used because it will not allocate a binding
            // partition, keeping this safepoint-free inside the serializer.
            jl_value_t *val = jl_get_latest_binding_value_if_resolved_and_const_debug_only(b);
            if (val == NULL || jl_is_module(val))
                continue;
            void **bp = ptrhash_bp(&extkey_paths, val);
            if (*bp != HT_NOTFOUND) {
                const extkey_path_t *old = (const extkey_path_t*)*bp;
                if (strcmp(jl_symbol_name(old->name), jl_symbol_name(b->globalref->name)) <= 0)
                    continue;   // keep the deterministic winner
            }
            extkey_path_t *e = (extkey_path_t*)malloc_s(sizeof(extkey_path_t));
            e->mod = b->globalref->mod;
            e->name = b->globalref->name;
            *bp = e;
        }
    }
}

// Drop the path table so the next build starts from its own module set. In the writer
// process one image is saved and the once-built table is right for its whole lifetime,
// but the loading process restores many images in one session: a table built from the
// first image's dependencies is missing every module only a later image depends on, so
// a `P:`-keyed object would resolve and then fail to re-render, refusing entries whose
// resolution was correct.
static void extkey_reset_paths(void) JL_NOTSAFEPOINT
{
    if (!extkey_paths_ready)
        return;
    for (size_t i = 0; i < extkey_paths.size; i += 2)
        if (extkey_paths.table[i + 1] != HT_NOTFOUND)
            free(extkey_paths.table[i + 1]);
    htable_free(&extkey_paths);
    extkey_paths_ready = 0;
}
// The chain of `UnionAll` binders in scope while a type is being keyed, innermost first.
// A bound `TypeVar` has no standalone identity -- only a meaning relative to the binder
// that introduced it -- so it is keyed by its de Bruijn index into this chain rather than
// by name. Names are gensyms as often as not (`#s42`) and are no part of a type's
// identity, so keying by them would be both unstable across rebuilds and gratuitously
// discriminating: `Vector{T} where T` and `Vector{S} where S` are one type and must
// produce one key.
typedef struct extkey_binder {
    struct extkey_binder *outer;
    jl_tvar_t *var;
} extkey_binder_t;

// Structural key for anything type-level: `DataType`, `Union`, `UnionAll`, `Vararg` and
// bound `TypeVar`s. Returns 0 for a type that cannot be keyed, which now means only one
// thing: it mentions a type variable no enclosing binder introduced.
static int extkey_type(ios_t *k, jl_value_t *t, extkey_binder_t *env, int depth,
                       int tdepth) JL_NOTSAFEPOINT;

// Every caller outside a binder chain enters at `depth` 0: a type reaches only types,
// symbols and boxed isbits values -- never back into a method, code instance or any other
// node of the value graph -- so it spends none of the value budget, and whatever the
// caller has left of it is irrelevant here. Type nesting is bounded separately, by
// `EXTKEY_MAX_TYPEDEPTH`.
#define extkey_type_toplevel(k, t) extkey_type(k, t, NULL, 0, 0)

// One string cannot be both the identity and the locator of an object, and trying to make
// it be both is what left the `TypeVar` keys unresolvable and the keys far larger than they
// need to be.
//
// The *identity* is what two builds compare: it must contain nothing build-specific, so it
// expands every sub-object in full, and it is what the digest is taken over. The *locator*
// is what this image reads back at load: it may cite an index into this image's own import
// table, exactly as every external reference already does, because that table is written
// and read by the same image. Locator mode is that second rendering; the identity is
// unchanged, which is why the stability measurements still hold.
static htable_t *extkey_import_index = NULL;   // jl_value_t* -> 1-based import index

static int extkey_import_ref(jl_value_t *v, size_t *out) JL_NOTSAFEPOINT
{
    if (extkey_import_index == NULL)
        return 0;
    void *p = ptrhash_get(extkey_import_index, v);
    if (p == HT_NOTFOUND)
        return 0;
    *out = (size_t)(uintptr_t)p;
    return 1;
}

// Digest of a source file's contents, cached per file symbol. Used to give method keys a
// body identity: without one, a rebuilt dependency whose method bodies changed but whose
// signatures did not would re-link silently, and dispatch-based revalidation cannot catch
// that because the dispatch answer is unchanged.
//
// This is deliberately coarse -- every method defined in a file shares its digest, so
// editing the file invalidates all of them -- but it is *stable*, which the compressed IR
// is not: that refers to method roots by index and root sets are keyed per build. Coarse
// and sound beats fine and unstable, because an unstable key silently fails to resolve and
// so buys no safety at all.
static htable_t extkey_filehash;
static int extkey_filehash_ready = 0;

static int extkey_file_digest(jl_sym_t *file, uint64_t *out) JL_NOTSAFEPOINT
{
    if (file == NULL)
        return 0;
    if (!extkey_filehash_ready) {
        htable_new(&extkey_filehash, 0);
        extkey_filehash_ready = 1;
    }
    void **bp = ptrhash_bp(&extkey_filehash, (void*)file);
    if (*bp != HT_NOTFOUND) {
        uint64_t h = *(uint64_t*)*bp;
        if (h == 0)
            return 0;   // remembered failure
        *out = h;
        return 1;
    }
    uint64_t h = 0;
    ios_t f;
    // Package methods carry absolute paths, which is the set that matters here; sysimage
    // methods carry bare relative names and are exempted by the caller.
    ios_t *fp = ios_file(&f, jl_symbol_name(file), 1, 0, 0, 0);
    if (fp != NULL) {
        h = 1469598103934665603ULL;   // FNV-1a over the file's bytes
        char buf[4096];
        size_t got;
        while ((got = ios_read(&f, buf, sizeof(buf))) > 0) {
            for (size_t i = 0; i < got; i++) {
                h ^= (unsigned char)buf[i];
                h *= 1099511628211ULL;
            }
            if (got < sizeof(buf))
                break;
        }
        ios_close(&f);
        if (h == 0)
            h = 1;   // never collide with the failure marker
    }
    uint64_t *slot = (uint64_t*)malloc_s(sizeof(uint64_t));
    *slot = h;
    *bp = slot;
    if (h == 0)
        return 0;
    *out = h;
    return 1;
}

// Reverse index from a bound type variable to a binder that introduces it. A `TypeVar`
// reached on its own -- as a method instance's static parameter, say -- has no identity of
// its own, but if some imported type binds it, then "the variable introduced by the binder
// at this depth of that type" names it stably. Built once per serialization.
typedef struct { uint64_t binder; int depth; jl_value_t *bobj; } extkey_tvar_t;
static htable_t extkey_tvars;
static int extkey_tvars_ready = 0;

static const extkey_tvar_t *extkey_lookup_tvar(jl_value_t *v) JL_NOTSAFEPOINT
{
    if (!extkey_tvars_ready)
        return NULL;
    void *p = ptrhash_get(&extkey_tvars, v);
    return p == HT_NOTFOUND ? NULL : (const extkey_tvar_t*)p;
}

// `ord` counts binders in pre-order within one root type. Nesting depth alone is not
// enough: two sibling binders at the same depth, as in `Tuple{Vector{T} where T,
// Vector{S} where S}`, would share it and merge two distinct variables. Pre-order position
// is structural, so it is identical in any build of the same type.
static void extkey_register_tvars(jl_value_t *t, uint64_t binder, jl_value_t *bobj, int *ord,
                                  int fuel) JL_NOTSAFEPOINT
{
    if (t == NULL || fuel <= 0)
        return;
    if (jl_is_unionall(t)) {
        jl_unionall_t *ua = (jl_unionall_t*)t;
        int here = (*ord)++;
        void **bp = ptrhash_bp(&extkey_tvars, (void*)ua->var);
        if (*bp == HT_NOTFOUND) {
            extkey_tvar_t *e = (extkey_tvar_t*)malloc_s(sizeof(extkey_tvar_t));
            e->binder = binder;
            e->depth = here;
            e->bobj = bobj;
            *bp = e;
        }
        else {
            // Deterministic when several roots bind one variable: keep the smaller
            // (binder, position) so two builds agree regardless of visit order.
            extkey_tvar_t *e = (extkey_tvar_t*)*bp;
            if (binder < e->binder || (binder == e->binder && here < e->depth)) {
                e->binder = binder;
                e->depth = here;
                e->bobj = bobj;
            }
        }
        extkey_register_tvars(ua->var->lb, binder, bobj, ord, fuel - 1);
        extkey_register_tvars(ua->var->ub, binder, bobj, ord, fuel - 1);
        extkey_register_tvars(ua->body, binder, bobj, ord, fuel - 1);
        return;
    }
    if (jl_is_uniontype(t)) {
        extkey_register_tvars(((jl_uniontype_t*)t)->a, binder, bobj, ord, fuel - 1);
        extkey_register_tvars(((jl_uniontype_t*)t)->b, binder, bobj, ord, fuel - 1);
        return;
    }
    if (jl_is_datatype(t)) {
        jl_svec_t *ps = ((jl_datatype_t*)t)->parameters;
        for (size_t i = 0; i < jl_svec_len(ps); i++)
            extkey_register_tvars(jl_svecref(ps, i), binder, bobj, ord, fuel - 1);
    }
}

// Raw bytes must never be written into a key directly: keys are NUL-terminated and
// compared as C strings, so an embedded zero -- which any integer value parameter has --
// would silently truncate the comparison and merge distinct keys.
static void extkey_bytes(ios_t *k, const char *p, size_t len) JL_NOTSAFEPOINT
{
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < len; i++) {
        ios_putc(hex[((unsigned char)p[i]) >> 4], k);
        ios_putc(hex[((unsigned char)p[i]) & 0xf], k);
    }
}

// Render one slot of a `CodeInstance.edges` vector. `edges` is a positional,
// heterogeneous encoding (callees, `Int` match counts that govern the following slots,
// bindings, invoke signatures, method-table markers), but a *digest* does not need to
// interpret it: rendering each slot in order is stable and discriminating as long as
// each slot renders stably. Callees are rendered shallowly -- recursing into their own
// edges would not terminate.
// Returns 0 if this slot has no stable rendering, which makes the whole edge digest --
// and therefore the owning code instance -- unkeyable. Rendering such a slot as a
// generic placeholder instead would silently merge distinct edge sets.
static int extkey_edge_slot(ios_t *k, jl_value_t *e) JL_NOTSAFEPOINT
{
    if (e == NULL) {
        ios_putc('0', k);
        return 1;
    }
    if (jl_is_code_instance(e)) {
        // Identify the callee by its method instance, not by which entry of that method
        // instance's cache chain this happens to point at. Recursing into the callee's
        // own edges would not terminate, and the sibling entries are equivalent anyway:
        // they agree on inferred code, return and exception types, effects and world
        // range, differing only in compilation state.
        jl_method_instance_t *mi = jl_get_ci_mi((jl_code_instance_t*)e);
        ios_puts("c<", k);
        if (!extkey_write(k, (jl_value_t*)mi, EXTKEY_MAX_DEPTH - 1))
            return 0;
        ios_putc('>', k);
        return 1;
    }
    if (jl_is_long(e)) {
        ios_printf(k, "i%zd", jl_unbox_long(e));
        return 1;
    }
    // Signatures and other type-valued slots key structurally, like every other type, but
    // at full depth: the budget above exists only to keep the recursion through callees
    // finite, and a type cannot reach back into a code instance.
    if (jl_is_type(e) || jl_is_vararg(e))
        return extkey_type_toplevel(k, e);
    return extkey_write(k, e, EXTKEY_MAX_DEPTH - 1);
}

// A 64-bit digest of a code instance's edge set, so the key stays bounded in size.
// Returns 0 if any slot lacks a stable rendering.
static int extkey_edges_hash(jl_code_instance_t *ci, uint64_t *out) JL_NOTSAFEPOINT
{
    jl_svec_t *edges = jl_atomic_load_relaxed(&ci->edges);
    if (edges == NULL) {
        *out = 0;
        return 1;
    }
    ios_t d;
    ios_mem(&d, 256);
    size_t n = jl_svec_len(edges);
    int ok = 1;
    for (size_t i = 0; i < n; i++) {
        if (!extkey_edge_slot(&d, jl_svecref(edges, i))) {
            ok = 0;
            break;
        }
        ios_putc('\x1e', &d);
    }
    if (ok) {
        uint64_t h = 1469598103934665603ULL;   // FNV-1a
        size_t len = (size_t)ios_pos(&d);
        for (size_t i = 0; i < len; i++) {
            h ^= (unsigned char)d.buf[i];
            h *= 1099511628211ULL;
        }
        *out = h;
    }
    ios_close(&d);
    return ok;
}

// `DebugInfo` has no name-based identity -- it is immutable line/file table data, and it
// is the single largest kind in the import table (26% of the distinct objects a large
// package imports). Key it by content instead. Returns 0 if any part lacks a stable
// rendering.
static int extkey_debuginfo(ios_t *k, jl_debuginfo_t *di, int depth) JL_NOTSAFEPOINT
{
    if (depth > EXTKEY_MAX_DEPTH)
        return 0;
    // def is a Method, MethodInstance or Symbol naming what this describes
    if (di->def != NULL && di->def != jl_nothing) {
        if (!extkey_write(k, di->def, depth + 1))
            return 0;
    }
    ios_putc(';', k);
    if (di->linetable != NULL && (jl_value_t*)di->linetable != jl_nothing) {
        if (!extkey_debuginfo(k, di->linetable, depth + 1))
            return 0;
    }
    ios_putc(';', k);
    // edges: declared as a SimpleVector but in practice a Memory{DebugInfo}; accept both
    jl_value_t *edges = (jl_value_t*)di->edges;
    if (edges != NULL && edges != jl_nothing) {
        size_t ne;
        jl_value_t **ed;
        if (jl_is_genericmemory(edges)) {
            ne = ((jl_genericmemory_t*)edges)->length;
            ed = jl_genericmemory_ptr_data((jl_genericmemory_t*)edges);
        }
        else if (jl_is_svec(edges)) {
            ne = jl_svec_len(edges);
            ed = jl_svec_data(edges);
        }
        else {
            return 0;
        }
        ios_printf(k, "n%zu", ne);
        for (size_t i = 0; i < ne; i++) {
            jl_value_t *e = ed[i];
            if (e == NULL)
                continue;
            if (!jl_typetagis(e, jl_debuginfo_type))
                return 0;
            if (!extkey_debuginfo(k, (jl_debuginfo_t*)e, depth + 1))
                return 0;
        }
    }
    ios_putc(';', k);
    // codelocs: compressed location data, hashed verbatim; either a String or a Memory
    if (di->codelocs == NULL || di->codelocs == jl_nothing) {
        /* nothing to add */
    }
    else if (jl_is_string(di->codelocs)) {
        ios_write(k, jl_string_data(di->codelocs), jl_string_len(di->codelocs));
    }
    else if (jl_is_genericmemory(di->codelocs)) {
        jl_genericmemory_t *m = (jl_genericmemory_t*)di->codelocs;
        const jl_datatype_layout_t *lo = ((jl_datatype_t*)jl_typeof(m))->layout;
        if (lo == NULL || lo->flags.arrayelem_isboxed)
            return 0;
        ios_write(k, (char*)m->ptr, m->length * lo->size);
    }
    else {
        return 0;
    }
    return 1;
}

// Names are length-prefixed rather than delimited. A generic function's type name is
// routinely `#@atomic`, `#>=` or `#<<`, and `@`, `<` and `>` are all structural in this
// grammar, so a delimited name cannot be read back at all -- measured at 59 of
// SparseArrays' keys, every one of them a macro or operator. The length also settles
// `var"a.b"`, which no amount of delimiter choice would have.
static void extkey_name(ios_t *k, jl_sym_t *s) JL_NOTSAFEPOINT
{
    const char *n = jl_symbol_name(s);
    size_t len = strlen(n);
    ios_printf(k, "%zu:", len);
    ios_write(k, n, len);
}

static void extkey_module(ios_t *k, jl_module_t *m) JL_NOTSAFEPOINT
{
    // root-first module path; the parent chain terminates at a root module
    jl_module_t *p = m->parent;
    if (p && p != m) {
        extkey_module(k, p);
        ios_putc('.', k);
    }
    extkey_name(k, m->name);
}

// A type parameter is either type-level (and must be keyed inside the binder
// environment) or an ordinary value such as a boxed integer or a symbol.
static int extkey_param(ios_t *k, jl_value_t *p, extkey_binder_t *env, int depth,
                        int tdepth) JL_NOTSAFEPOINT
{
    if (p == NULL)
        return 0;
    if (jl_is_type(p) || jl_is_typevar(p) || jl_is_vararg(p))
        return extkey_type(k, p, env, depth, tdepth);
    return extkey_write(k, p, depth + 1);
}

// `Union` is a binary tree over a set of components, but `jl_type_union` canonicalizes
// that tree -- it sorts and de-duplicates -- so the flattened in-order sequence is stable.
static size_t extkey_union_count(jl_value_t *t) JL_NOTSAFEPOINT
{
    if (jl_is_uniontype(t))
        return extkey_union_count(((jl_uniontype_t*)t)->a) +
               extkey_union_count(((jl_uniontype_t*)t)->b);
    return 1;
}

static int extkey_union_parts(ios_t *k, jl_value_t *t, extkey_binder_t *env, int depth,
                              int tdepth, int *first) JL_NOTSAFEPOINT
{
    if (jl_is_uniontype(t))
        return extkey_union_parts(k, ((jl_uniontype_t*)t)->a, env, depth, tdepth, first) &&
               extkey_union_parts(k, ((jl_uniontype_t*)t)->b, env, depth, tdepth, first);
    if (!*first)
        ios_putc(',', k);
    *first = 0;
    return extkey_type(k, t, env, depth, tdepth);
}

static int extkey_type(ios_t *k, jl_value_t *t, extkey_binder_t *env, int depth,
                       int tdepth) JL_NOTSAFEPOINT
{
    if (t == NULL || depth > EXTKEY_MAX_DEPTH || tdepth > EXTKEY_MAX_TYPEDEPTH)
        return 0;
    // Most of a key is nested types, so this is where citing an import instead of
    // expanding it actually pays. Only outside a binder, though: a type carrying free
    // variables means them relative to the binder being rebuilt around it, and an entry
    // resolved by index would carry the original variables instead.
    // Never cite a bare type variable: resolving one rebuilds a fresh variable rather than
    // finding the original, so the citing type comes back with a different variable in it --
    // measured as `DenseArray{T, 1}` resolving to `DenseArray{_, 1}`, six times in
    // SparseArrays. A type that merely *contains* free variables is safe, because citing it
    // finds that object itself.
    if (tdepth > 0 && env == NULL && !jl_is_typevar(t)) {
        size_t idx;
        if (extkey_import_ref(t, &idx)) {
            ios_printf(k, "@%zu", idx);
            return 1;
        }
    }
    if (jl_is_typevar(t)) {
        int i = 0;
        for (extkey_binder_t *b = env; b != NULL; b = b->outer, i++) {
            if (b->var == (jl_tvar_t*)t) {
                ios_printf(k, "#%d", i);
                return 1;
            }
        }
        return 0;   // free type variable: nothing binds it, so it has no identity here
    }
    if (t == (jl_value_t*)jl_bottom_type) {
        ios_puts("U0<>", k);   // `Union{}` is the empty union
        return 1;
    }
    if (jl_is_uniontype(t)) {
        int first = 1;
        ios_printf(k, "U%zu<", extkey_union_count(t));
        if (!extkey_union_parts(k, t, env, depth, tdepth + 1, &first))
            return 0;
        ios_putc('>', k);
        return 1;
    }
    if (jl_is_unionall(t)) {
        // the bounds are keyed outside the binder, the body inside it
        jl_unionall_t *ua = (jl_unionall_t*)t;
        ios_puts("A<", k);
        if (!extkey_type(k, ua->var->lb, env, depth, tdepth + 1))
            return 0;
        ios_putc(',', k);
        if (!extkey_type(k, ua->var->ub, env, depth, tdepth + 1))
            return 0;
        ios_putc(';', k);
        extkey_binder_t b = { env, ua->var };
        if (!extkey_type(k, ua->body, &b, depth, tdepth + 1))
            return 0;
        ios_putc('>', k);
        return 1;
    }
    if (jl_is_vararg(t)) {
        // both fields are optional: `Vararg`, `Vararg{T}` and `Vararg{T,N}` all occur
        jl_vararg_t *vm = (jl_vararg_t*)t;
        ios_puts("X<", k);
        if (vm->T == NULL)
            ios_putc('-', k);
        else if (!extkey_type(k, vm->T, env, depth, tdepth + 1))
            return 0;
        ios_putc(',', k);
        if (vm->N == NULL)
            ios_putc('-', k);
        else if (!extkey_param(k, vm->N, env, depth, tdepth + 1))
            return 0;   // a bound length variable, or a boxed `Int`
        ios_putc('>', k);
        return 1;
    }
    if (jl_is_datatype(t)) {
        jl_datatype_t *dt = (jl_datatype_t*)t;
        ios_puts("T:", k);
        extkey_module(k, dt->name->module);
        ios_putc('.', k);
        extkey_name(k, dt->name->name);
        // parameters are part of the identity of an instantiated type
        size_t np = jl_svec_len(dt->parameters);
        // An empty parameter list is not the same as no parameter list: `Tuple{}` and
        // `Tuple` are different types that both have zero parameters, and keying them
        // identically merged them -- found by parsing keys back and comparing. What tells
        // them apart is that an instantiation is not its own type name's wrapper.
        if (np == 0 && dt->name->wrapper != (jl_value_t*)dt)
            ios_puts("{0:}", k);
        if (np) {
            ios_printf(k, "{%zu:", np);
            for (size_t i = 0; i < np; i++) {
                if (i) ios_putc(',', k);
                if (!extkey_param(k, jl_svecref(dt->parameters, i), env, depth, tdepth + 1))
                    return 0;
            }
            ios_putc('}', k);
        }
        return 1;
    }
    return 0;
}

// One string cannot be both the identity and the locator of an object, and trying to make
// it be both is what left the `TypeVar` keys unresolvable and the keys ten times larger
// than they need to be.
//
// The *identity* is what two builds compare: it must contain nothing build-specific, so it
// expands every sub-object in full and is what the digest is taken over. The *locator* is
// what this image reads back at load: it may cite an index into this image's own import
// table, exactly as every external reference already does, because that table is written
// and read by the same image. Locator mode is that second rendering, and it is enabled
// only while the import table is being written.
static int extkey_write(ios_t *k, jl_value_t *v, int depth) JL_NOTSAFEPOINT
{
    if (depth > EXTKEY_MAX_DEPTH)
        return 0;
    if (depth > 0 && !jl_is_typevar(v)) {
        // In locator mode a nested object that is itself imported is cited rather than
        // expanded. At depth 0 this would make every entry a reference to itself.
        size_t idx;
        if (extkey_import_ref(v, &idx)) {
            ios_printf(k, "@%zu", idx);
            return 1;
        }
    }
    if (jl_is_module(v)) {
        ios_puts("M:", k);
        extkey_module(k, (jl_module_t*)v);
        return 1;
    }
    if (jl_is_typename(v)) {
        jl_typename_t *tn = (jl_typename_t*)v;
        ios_puts("N:", k);
        extkey_module(k, tn->module);
        ios_putc('.', k);
        extkey_name(k, tn->name);
        return 1;
    }
    if (jl_is_datatype(v) || jl_is_type(v) || jl_is_vararg(v)) {
        // All type-level kinds are keyed structurally, binders and all, so that the key
        // records what a type *is* rather than how it prints. A bare `TypeVar` is not
        // included: it is only meaningful relative to the `UnionAll` that binds it, and
        // `extkey_type` refuses one that no enclosing binder introduced.
        return extkey_type_toplevel(k, v);
    }
    if (jl_is_binding(v)) {
        jl_binding_t *b = (jl_binding_t*)v;
        ios_puts("B:", k);
        extkey_module(k, b->globalref->mod);
        ios_putc('.', k);
        extkey_name(k, b->globalref->name);
        return 1;
    }
    if (jl_is_method(v)) {
        jl_method_t *m = (jl_method_t*)v;
        ios_puts("F:", k);
        extkey_module(k, m->module);
        ios_putc('.', k);
        extkey_name(k, m->name);
        ios_putc('@', k);
        // The signature disambiguates the methods of one generic function. Deliberately
        // *not* including file:line -- `Method.file` is an absolute path into the depot,
        // which differs between machines and depots and would make keys unportable.
        if (!extkey_type_toplevel(k, (jl_value_t*)m->sig))
            return 0;
        // Module, name and signature identify a method's *definition site*, not its body,
        // and that is not enough. Code inferred against a method -- possibly with it
        // inlined -- must not re-link to a rebuilt version whose body changed while its
        // signature did not: dispatch-based revalidation cannot catch that, because the
        // dispatch answer is unchanged, so it would silently run stale code. Bumping a
        // dependency's patch version is exactly that case.
        //
        // The body identity is the defining file's contents. Digesting `Method.source`
        // instead was tried and reverted: compressed IR refers to method roots by index
        // and root sets are keyed per build, so it is not byte-stable across independent
        // builds of identical source -- on Makie it perturbed 4127 keys. Refusing a key
        // when the file cannot be read keeps this conservative: no key means no re-link,
        // which is exactly today's behaviour.
        // Only methods in a *rebuildable* image need a body identity. A method in the
        // sysimage cannot change without changing the Julia build, which already
        // invalidates every cache through the header check, and its `file` is a bare
        // relative name (`reduce.jl`) that cannot be resolved from the symbol anyway.
        // Demanding a digest there fails, and the failure cascades into every pkgimage
        // code instance that calls into Base.
        size_t blob = external_blob_index((jl_value_t*)m);
        if (blob != 0 && blob < n_linkage_blobs()) {
            uint64_t fh;
            if (!extkey_file_digest(m->file, &fh))
                return 0;
            ios_printf(k, "@F%016" PRIx64, fh);
        }
        return 1;
    }
    if (jl_is_method_instance(v)) {
        jl_method_instance_t *mi = (jl_method_instance_t*)v;
        if (!jl_is_method(mi->def.value))
            return 0;   // toplevel thunks have no stable name
        ios_puts("I:", k);
        if (!extkey_write(k, mi->def.value, depth + 1))
            return 0;
        ios_putc('/', k);
        return extkey_type_toplevel(k, (jl_value_t*)mi->specTypes);
    }
    if (jl_is_code_instance(v)) {
        jl_code_instance_t *ci = (jl_code_instance_t*)v;
        jl_method_instance_t *mi = jl_get_ci_mi(ci);
        ios_puts("C:", k);
        if (!extkey_write(k, (jl_value_t*)mi, depth + 1))
            return extkey_ci_fail[EK_CI_MI]++, 0;
        ios_putc('/', k);
        // the owner distinguishes foreign-interpreter caches sharing one MethodInstance
        if (ci->owner == jl_nothing)
            ios_putc('-', k);
        else if (!extkey_write(k, ci->owner, depth + 1))
            return extkey_ci_fail[EK_CI_OWNER]++, 0;
        // and the ABI/rettype distinguishes co-existing entries for one owner
        ios_putc('/', k);
        if (!extkey_type_toplevel(k, ci->rettype))
            return extkey_ci_fail[EK_CI_RETTYPE]++, 0;
        // Entries in one method instance's cache chain can agree on all of the above and
        // differ only in their edges, so the edge set has to take part in the identity.
        // The edge digest is identity data even inside a locator: it is a hash, so it can
        // only ever be compared, and it is compared against one recomputed from a live
        // candidate with no import table in hand. Rendering the edges with references on
        // made those two disagree for 65 of SparseArrays' code instances -- every one of
        // them a candidate that matched on every other field.
        uint64_t ehash;
        htable_t *saved_refs = extkey_import_index;
        extkey_import_index = NULL;
        int edges_ok = extkey_edges_hash(ci, &ehash);
        extkey_import_index = saved_refs;
        if (!edges_ok)
            return extkey_ci_fail[EK_CI_EDGES]++, 0;
        ios_printf(k, "/E%016" PRIx64, ehash);
        // The remaining inference results a caller can have specialized against. Omitting
        // any of them merges code instances that are not interchangeable: a caller that
        // inlined one and elided a branch on its effects, or constant-folded through its
        // `rettype_const`, is not correct against the other.
        ios_putc('/', k);
        if (!extkey_type_toplevel(k, ci->exctype))
            return extkey_ci_fail[EK_CI_EXCTYPE]++, 0;
        ios_putc('/', k);
        if (ci->rettype_const == NULL)
            ios_putc('-', k);
        else if (!extkey_write(k, ci->rettype_const, depth + 1))
            // a constant we cannot name is a constant we cannot re-link against
            return extkey_ci_fail[EK_CI_RETCONST]++, extkey_note_retconst(ci->rettype_const), 0;
        ios_printf(k, "/P%08" PRIx32, jl_atomic_load_relaxed(&ci->ipo_purity_bits));
        // World ages themselves are per-build counters and would never match across a
        // rebuild, but they are not serialized raw either: an incremental image collapses
        // every code instance to one of two states on write (staticdata.c, `jl_is_code_instance`
        // in the fixup pass) -- live, and so subject to revalidation, or dead. That
        // distinction is the only part of the world range that survives, so it is the only
        // part the key can carry.
        ios_putc(jl_atomic_load_relaxed(&ci->max_world) == ~(size_t)0 ? 'L' : 'D', k);
        // Deliberately excluded: `analysis_results` (a derived cache with no stable
        // identity of its own; the effects it summarizes are already in the purity bits),
        // and every compilation-state field -- `invoke`, `specptr`, `precompile`, the
        // `time_infer_*` counters -- which record what this build happened to compile
        // rather than what was inferred.
        return 1;
    }
    if (jl_is_symbol(v)) {
        ios_puts("S:", k);
        extkey_name(k, (jl_sym_t*)v);
        return 1;
    }
    if (jl_is_string(v)) {
        ios_puts("s:", k);
        extkey_bytes(k, jl_string_data(v), jl_string_len(v));
        return 1;
    }
    if (jl_typetagis(v, jl_debuginfo_type)) {
        // digest the content so the key stays bounded
        ios_t d;
        ios_mem(&d, 256);
        int ok = extkey_debuginfo(&d, (jl_debuginfo_t*)v, depth);
        if (ok) {
            uint64_t h = 1469598103934665603ULL;   // FNV-1a
            size_t len = (size_t)ios_pos(&d);
            for (size_t i = 0; i < len; i++) {
                h ^= (unsigned char)d.buf[i];
                h *= 1099511628211ULL;
            }
            ios_printf(k, "D:%016" PRIx64, h);
        }
        ios_close(&d);
        return ok;
    }
    if (jl_is_svec(v)) {
        // element-wise; svecs in the import table are mostly type tuples
        ios_puts("V:", k);
        size_t n = jl_svec_len(v);
        ios_printf(k, "%zu<", n);
        for (size_t i = 0; i < n; i++) {
            jl_value_t *e = jl_svecref(v, i);
            if (i)
                ios_putc(',', k);
            if (e == NULL)
                ios_putc('0', k);
            else if (!extkey_write(k, e, depth + 1))
                return 0;
        }
        ios_putc('>', k);
        return 1;
    }
    if (jl_is_globalref(v)) {
        // A global reference is exactly the module and name it names.
        jl_globalref_t *g = (jl_globalref_t*)v;
        ios_puts("R:", k);
        extkey_module(k, g->mod);
        ios_putc('.', k);
        extkey_name(k, g->name);
        return 1;
    }
    if (jl_is_genericmemory(v)) {
        // No name; content, exactly like DebugInfo. The layout on a GenericMemory's type
        // describes its element, so boxed elements recurse and inline ones hash their
        // bytes -- but only when those bytes are fully defined, hence the padding guard.
        jl_genericmemory_t *m = (jl_genericmemory_t*)v;
        const jl_datatype_layout_t *lo = ((jl_datatype_t*)jl_typeof(m))->layout;
        if (lo == NULL)
            return 0;
        ios_puts("G:", k);
        if (!extkey_type_toplevel(k, (jl_value_t*)jl_typeof(m)))
            return 0;
        ios_printf(k, "<%zu:", (size_t)m->length);
        if (lo->flags.arrayelem_isboxed) {
            jl_value_t **el = jl_genericmemory_ptr_data(m);
            for (size_t i = 0; i < (size_t)m->length; i++) {
                if (i)
                    ios_putc(',', k);
                if (el[i] == NULL)
                    ios_putc('0', k);
                else if (!extkey_write(k, el[i], depth + 1))
                    return 0;
            }
        }
        else if (lo->npointers == 0 && !lo->flags.haspadding) {
            extkey_bytes(k, (char*)m->ptr, (size_t)m->length * lo->size);
        }
        else {
            return 0;   // inline elements carrying pointers or padding
        }
        ios_putc('>', k);
        return 1;
    }
    if (jl_is_mtable(v)) {
        // Method tables are only ever created for the global `Core.methodtable`, as a
        // clone carrying an existing table's name and module, or by
        // `Base.Experimental.@MethodTable name`, which always binds the result to a
        // constant. There is no anonymous construction path, so name and module identify
        // one exactly.
        jl_methtable_t *mt = (jl_methtable_t*)v;
        ios_puts("MT:", k);
        extkey_module(k, mt->module);
        ios_putc('.', k);
        extkey_name(k, mt->name);
        return 1;
    }
    jl_datatype_t *vt = (jl_datatype_t*)jl_typeof(v);
    if (jl_is_datatype_singleton(vt)) {
        // A singleton -- `nothing`, `Colon()`, `IndexLinear()`, a generic function object,
        // a closure with no captures -- has exactly one instance, so its type identifies
        // it completely. This also covers the long tail of `#foo`-typed function objects.
        ios_puts("O:", k);
        return extkey_write(k, (jl_value_t*)vt, depth + 1);
    }
    if (jl_is_datatype(vt) && vt->layout != NULL && vt->layout->npointers == 0 &&
        !vt->layout->flags.haspadding && jl_is_immutable(vt)) {
        // A boxed immutable with no pointers and no padding -- an Int, Bool, Char, Float,
        // or a tightly packed isbits struct used as a type parameter, such as the `3` in
        // `NTuple{3,Float64}` -- is exactly its type plus its bytes. Padding is excluded
        // because gap bytes are not required to be initialized, so two logically identical
        // values could otherwise key differently within a single build. Without this,
        // every parameterized type carrying a value parameter was unkeyable.
        ios_puts("b:", k);
        if (!extkey_write(k, (jl_value_t*)vt, depth + 1))
            return 0;
        ios_putc('<', k);
        extkey_bytes(k, (const char*)v, jl_datatype_size(vt));
        ios_putc('>', k);
        return 1;
    }
    if (jl_is_datatype(vt) && jl_is_immutable(vt) && vt->layout != NULL &&
        !vt->name->abstract && jl_datatype_nfields(vt) > 0) {
        // A general immutable struct is its type plus its fields: pointer fields recurse,
        // inline fields contribute their bytes. This generalizes the pointer-free case
        // above and covers wrappers such as `Const` and mixed structs like
        // `VersionNumber`. Only immutables qualify -- a mutable object's contents can
        // change after serialization, so no content key for one could be stable.
        size_t nf = jl_datatype_nfields(vt);
        ios_puts("v:", k);
        if (!extkey_write(k, (jl_value_t*)vt, depth + 1))
            return 0;
        ios_putc('{', k);
        for (size_t i = 0; i < nf; i++) {
            if (i)
                ios_putc(',', k);
            if (jl_field_isptr(vt, i)) {
                jl_value_t *fv = jl_get_nth_field_noalloc(v, i);
                if (fv == NULL)
                    ios_putc('0', k);
                else if (!extkey_write(k, fv, depth + 1))
                    return 0;
            }
            else {
                // An inline field's bytes are only meaningful if they hold no pointers,
                // which would be addresses rather than content, and no padding.
                jl_value_t *ft = jl_field_type_concrete(vt, i);
                if (!jl_is_datatype(ft))
                    return 0;
                const jl_datatype_layout_t *flo = ((jl_datatype_t*)ft)->layout;
                if (flo == NULL || flo->npointers != 0 || flo->flags.haspadding)
                    return 0;
                extkey_bytes(k, (const char*)v + jl_field_offset(vt, i), jl_field_size(vt, i));
            }
        }
        ios_putc('}', k);
        return 1;
    }
    if (jl_is_typevar(v)) {
        // A type variable reached on its own has no identity, but if some imported type
        // binds it then that binder plus the depth at which it is introduced names it.
        const extkey_tvar_t *tv = extkey_lookup_tvar(v);
        if (tv != NULL) {
            // The identity names the binder by digest, which is stable but cannot be
            // inverted; the locator cites the binder's import entry, which can. This is
            // the one place where the two renderings say genuinely different things
            // rather than the same thing at different lengths.
            size_t idx;
            if (tv->bobj != NULL && extkey_import_ref(tv->bobj, &idx)) {
                ios_printf(k, "TV@%zu/%d", idx, tv->depth);
                return 1;
            }
            ios_printf(k, "TV:%016" PRIx64 "/%d", tv->binder, tv->depth);
            return 1;
        }
        return 0;   // nothing imported binds it
    }
    // Last resort: an object with no identity of its own may still be reachable by a
    // stable path. A mutable global -- a `LazyLibrary` in a JLL, a lock, a cache -- is
    // typically the value of a module constant, and that binding *is* stably named. Key
    // it by where it lives rather than by what it contains. `extkey_binding_path` is
    // populated once per serialization by scanning the loaded modules' binding tables.
    {
        const extkey_path_t *p = extkey_lookup_path(v);
        if (p != NULL) {
            ios_puts("P:", k);
            extkey_module(k, p->mod);
            ios_putc('.', k);
            extkey_name(k, p->name);
            return 1;
        }
    }
    // What is left genuinely has no stable identity: bare TypeVars, which mean nothing
    // apart from the binder that scopes them, and unreachable mutable state.
    return 0;
}

// Digest an object's content key to 64 bits. Returns 0 if the object has no stable key,
// in which case a rebuilt dependency could not be re-resolved for it and the importing
// image would have to be rebuilt, as it always is today.
static int extkey_hash(jl_value_t *v, uint64_t *out) JL_NOTSAFEPOINT
{
    ios_t k;
    ios_mem(&k, 128);
    int ok = extkey_write(&k, v, 0);
    if (ok) {
        uint64_t h = 1469598103934665603ULL;   // FNV-1a
        size_t len = (size_t)ios_pos(&k);
        for (size_t i = 0; i < len; i++) {
            h ^= (unsigned char)k.buf[i];
            h *= 1099511628211ULL;
        }
        *out = h;
    }
    ios_close(&k);
    return ok;
}

// Does a key name both of these? A key identifies an *equivalence class*, not an
// allocation: distinct allocations that Julia itself treats as one identity are what the
// loader is free to merge. Type uniquing merges equal types on load, `jl_egal` is Julia's
// own definition of "the same value" (covering separately allocated but equal simple
// vectors, strings and debug info), and sibling entries in one method instance's cache
// chain are interchangeable. Used both to excuse key collisions and to judge whether a
// shadow-resolved object is the object we started from.
static int extkey_equiv(jl_value_t *oa, jl_value_t *ob) JL_NOTSAFEPOINT
{
    if (oa == ob)
        return 1;
    if (oa == NULL || ob == NULL)
        return 0;
    if (jl_egal(oa, ob))
        return 1;
    if (jl_is_type(oa) && jl_is_type(ob) && jl_types_equal(oa, ob))
        return 1;
    if (jl_is_typevar(oa) && jl_is_typevar(ob)) {
        // Two structurally identical types are often separate allocations, and each binds
        // its own variable objects. Those are the same variable: a type variable's name is
        // a gensym as often as not and carries no identity, so compare what does -- the
        // bounds. Without this, keying variables by their binder reports one collision per
        // duplicated binder.
        jl_tvar_t *ta = (jl_tvar_t*)oa, *tb = (jl_tvar_t*)ob;
        if (jl_types_equal(ta->lb, tb->lb) && jl_types_equal(ta->ub, tb->ub))
            return 1;
    }
    if (jl_is_code_instance(oa) && jl_is_code_instance(ob)) {
        jl_code_instance_t *ca = (jl_code_instance_t*)oa;
        jl_code_instance_t *cb = (jl_code_instance_t*)ob;
        if (jl_get_ci_mi(ca) == jl_get_ci_mi(cb) && ca->owner == cb->owner &&
            ca->rettype == cb->rettype && ca->exctype == cb->exctype &&
            ca->rettype_const == cb->rettype_const &&
            jl_atomic_load_relaxed(&ca->min_world) == jl_atomic_load_relaxed(&cb->min_world) &&
            jl_atomic_load_relaxed(&ca->max_world) == jl_atomic_load_relaxed(&cb->max_world))
            return 1;
    }
    return 0;
}

// Drop the type-variable table so the next build starts from its own object set. The
// table is scoped to one image's imports -- the writer's at save, one loaded table's at
// relink -- and letting a previous image's registrations leak into the next would move
// the deterministic (binder, position) winners and change the keys.
static void extkey_reset_tvars(void) JL_NOTSAFEPOINT
{
    if (!extkey_tvars_ready)
        return;
    for (size_t i = 0; i < extkey_tvars.size; i += 2)
        if (extkey_tvars.table[i + 1] != HT_NOTFOUND)
            free(extkey_tvars.table[i + 1]);
    htable_free(&extkey_tvars);
    extkey_tvars_ready = 0;
}

// Register every type variable bound by any imported type, so that a type variable
// reached on its own can still be named. Must run before any key is computed.
static void extkey_build_tvars(jl_serializer_state *s) JL_NOTSAFEPOINT
{
    extkey_reset_tvars();
    htable_new(&extkey_tvars, 0);
    extkey_tvars_ready = 1;
    for (size_t i = 0; i < s->import_objs.len; i++) {
        jl_value_t *v = (jl_value_t*)s->import_objs.items[i];
        if (!jl_is_type(v))
            continue;
        // The binder is identified by its own structural key, which is already stable.
        uint64_t h = 0;
        if (extkey_hash(v, &h))
            { int ord = 0; extkey_register_tvars(v, h, v, &ord, EXTKEY_MAX_TYPEDEPTH); }
    }
}

// Serialize the import table: for each distinct object this image references in another
// image, the owning image's deps-index and the object's content key, with a zero key
// marking an object that has no stable identity. Nothing reads this back yet beyond
// checking that it round-trips; resolving through it is what would let a rebuilt
// dependency be re-linked instead of invalidating every image above it.
// REVIEW instrumentation for the identity-hash hazard, which is a precondition for any
// re-linking scheme rather than a property of the keys. `jl_object_id` of a mutable object
// is not derived from its content: the serializer bakes the id the object had in the
// writing process into the header ahead of it (`object_id_expected`, below), and
// `jl_object_id__cold` reads that back for anything tagged `GC_IN_IMAGE`
// (src/builtins.c:377,490). A rebuilt dependency therefore hands out *different* ids for
// content-identical objects.
//
// So any table this image serialized whose slot layout was computed from such an id -- an
// `IdDict`, or a `Dict`/`Set` whose key type falls back to the generic `objectid` hash --
// would silently mis-look-up after being re-linked against a rebuilt dependency, and there
// is no rehash pass on the restore path to repair it (only the type cache rehashes,
// `cache_rehash_set`). Lookups miss and entries duplicate: no crash, no error.
//
// This measures the exposure. `containers` is the population a conservative "refuse to
// re-link an image that serializes one" rule would have to consider; `tainted` is the
// subset that actually reaches a mutable owned by another image, which is the only part at
// risk. If `tainted` is near zero across real packages, refusal is free and the hazard
// costs nothing to close.
// A mutable type whose `objectid` is nonetheless derived from content, and so survives a
// rebuild. `jl_object_id__cold` (src/builtins.c) special-cases exactly these, and the
// serializer's `object_id_expected` predicate excludes exactly the same set from having an
// id baked ahead of it -- the two lists are the same fact seen from either side.
static int idhash_content_hashed(jl_value_t *t) JL_NOTSAFEPOINT
{
    return t == (jl_value_t*)jl_string_type || t == (jl_value_t*)jl_symbol_type ||
           t == (jl_value_t*)jl_simplevector_type || t == (jl_value_t*)jl_datatype_type ||
           t == (jl_value_t*)jl_module_type || t == (jl_value_t*)jl_typename_type;
}

static int idhash_hit(jl_value_t *el) JL_NOTSAFEPOINT
{
    if (el == NULL)
        return 0;
    size_t blob = external_blob_index(el);
    if (blob >= n_linkage_blobs())
        return 0;
    jl_value_t *t = jl_typeof(el);
    return jl_is_datatype(t) && jl_is_mutable(t) && !idhash_content_hashed(t);
}

static size_t idhash_scan(jl_value_t *v, int depth) JL_NOTSAFEPOINT
{
    if (v == NULL || depth > 3)
        return 0;
    if (idhash_hit(v))
        return 1;   // an external mutable: stop here, its contents are not ours
    if (external_blob_index(v) < n_linkage_blobs())
        return 0;
    size_t hits = 0;
    if (jl_is_genericmemory(v)) {
        jl_genericmemory_t *m = (jl_genericmemory_t*)v;
        const jl_datatype_layout_t *lo = ((jl_datatype_t*)jl_typeof(m))->layout;
        if (lo != NULL && lo->flags.arrayelem_isboxed) {
            jl_value_t **el = jl_genericmemory_ptr_data(m);
            for (size_t i = 0; i < (size_t)m->length; i++)
                hits += idhash_scan(el[i], depth + 1);
        }
        return hits;
    }
    jl_datatype_t *dt = (jl_datatype_t*)jl_typeof(v);
    if (!jl_is_datatype(dt) || dt->layout == NULL)
        return 0;
    for (size_t f = 0; f < jl_datatype_nfields(dt); f++) {
        if (!jl_field_isptr(dt, f))
            continue;
        jl_value_t *fv = *(jl_value_t**)((char*)v + jl_field_offset(dt, f));
        hits += idhash_scan(fv, depth + 1);
    }
    return hits;
}

static void jl_report_idhash_taint(jl_serializer_state *s) JL_NOTSAFEPOINT
{
    (void)s;
    size_t n_id = 0, n_hash = 0, n_id_tainted = 0, n_hash_tainted = 0, n_hits = 0;
    // histogram by concrete container type: whether a rehash-on-relink pass is bounded
    // work or open-ended depends on whether the tainted population is a handful of Base
    // container types or an open set of user types.
    enum { TTY_MAX = 128 };
    void *tty[TTY_MAX]; size_t tcnt[TTY_MAX], ntty = 0, tdropped = 0;
    for (size_t i = 0; i < serialization_queue.len; i++) {
        jl_value_t *v = (jl_value_t*)serialization_queue.items[i];
        if (v == NULL || v == (jl_value_t*)(uintptr_t)-1 || v == (jl_value_t*)(uintptr_t)-2)
            continue;
        jl_value_t *t = jl_typeof(v);
        if (!jl_is_datatype(t))
            continue;
        const char *tn = jl_symbol_name(((jl_datatype_t*)t)->name->name);
        // `IdDict`/`IdSet` hash by `objectid` unconditionally. `Dict`/`Set` hash by
        // `hash(key)`, which falls back to `objectid` for any mutable key type with no
        // method of its own -- undecidable here, so they are counted separately rather
        // than merged into one alarming number.
        int identity = !strcmp(tn, "IdDict") || !strcmp(tn, "IdSet") || !strcmp(tn, "WeakKeyIdDict");
        int hashed = !identity && (!strcmp(tn, "Dict") || !strcmp(tn, "Set") || !strcmp(tn, "WeakKeyDict"));
        if (!identity && !hashed)
            continue;
        // Only the *key* type decides the hazard. `Dict{Symbol,Any}` is not exposed however
        // many external mutables sit on its value side: a Symbol is interned and hashes by
        // its name. A key type that is abstract, or concrete and mutable, is the case where
        // `hash` falls back to `objectid` and the slot layout depends on an address from
        // the writing process.
        jl_svec_t *par = ((jl_datatype_t*)t)->parameters;
        jl_value_t *K = jl_svec_len(par) > 0 ? jl_svecref(par, 0) : NULL;
        int keyed_by_id = identity ||
            (K != NULL && (!jl_is_datatype(K) || !jl_is_concrete_type(K) ||
                           (jl_is_mutable(K) && !idhash_content_hashed(K))));
        if (!keyed_by_id)
            continue;
        size_t hits = idhash_scan(v, 0);
        if (identity) {
            n_id++;
            if (hits) n_id_tainted++;
        }
        else {
            n_hash++;
            if (hits) n_hash_tainted++;
        }
        n_hits += hits;
        if (hits) {
            size_t q;
            for (q = 0; q < ntty; q++)
                if (tty[q] == (void*)t) break;
            if (q == ntty) {
                if (q == TTY_MAX) { tdropped++; continue; }
                tty[q] = (void*)t; tcnt[q] = 0; ntty++;
            }
            tcnt[q]++;
        }
    }
    for (size_t q = 0; q < ntty; q++) {
        jl_safe_printf("IDHASH_TAINT %6zu ", tcnt[q]);
        jl_static_show(JL_STDERR, (jl_value_t*)tty[q]);
        jl_safe_printf("\n");
    }
    if (tdropped)
        jl_safe_printf("IDHASH_TAINT <overflow> %zu\n", tdropped);
    jl_safe_printf("IDHASH containers_id=%zu tainted_id=%zu containers_hash=%zu tainted_hash=%zu external_mutable_refs=%zu\n",
                   n_id, n_id_tainted, n_hash, n_hash_tainted, n_hits);
}

static void jl_write_import_table(jl_serializer_state *s, ios_t *f) JL_NOTSAFEPOINT
{
    size_t n = s->import_objs.len;
    write_uint32(f, (uint32_t)n);
    ios_t k;
    ios_mem(&k, 4096);
    size_t keybytes = 0;
    // An object the writer cannot render carries no locator, so no loader can re-derive
    // it and its dependency edge can never be re-linked -- which makes the unkeyed
    // population, not the failing one, the ceiling on whole-edge acceptance. Only the
    // writer can name these: at load the recorded offset is meaningless against a
    // rebuilt blob.
#define UK_MAX 24
#define UK_NEX 2
    const char *uk_name[UK_MAX];
    size_t uk_count[UK_MAX];
    jl_value_t *uk_ex[UK_MAX][UK_NEX];
    size_t uk_nex[UK_MAX];
    int nuk = 0;
    size_t uk_wrap[4] = {0, 0, 0, 0};   // wrapper body / neither / bare typevar / own-wrapper vars
    for (size_t i = 0; i < n; i++) {
        jl_value_t *v = (jl_value_t*)s->import_objs.items[i];
        uint64_t h = 0;
        extkey_hash(v, &h);
        // The deps-index, not the blob index: this is the same number `add_external_linkage`
        // packs into every reference to `v`, and it is what a loader can recompute. The
        // blob index is a per-process load order that means nothing in another session.
        size_t blobidx = (size_t)(uintptr_t)s->import_deps.items[i];
        uint32_t depsidx = (uint32_t)jl_array_data(s->buildid_depmods_idxs, int32_t)[blobidx];
        write_uint32(f, depsidx);
        write_uint64(f, h);
        // The offset this object has in its owning blob, which is what every reference to
        // it in this image carries. Recording it is what lets a relink find the entry a
        // reference belongs to without changing how references are encoded: the loader
        // maps (deps-index, offset) to the object it resolved, and only on the slow path.
        size_t blob = external_blob_index(v);
        uint64_t off = 0;
        if (blob < n_linkage_blobs())
            off = ((uintptr_t)v - (uintptr_t)jl_linkage_blobs.items[2 * blob]) / SYS_EXTERNAL_LINK_UNIT;
        write_uint64(f, off);
        // The digest alone cannot be turned back into the object it names -- it is not
        // invertible -- so resolution against a rebuilt dependency needs the key itself.
        // The digest stays as the *verification*: resolution finds a candidate through the
        // runtime tables by structure, recomputes its key, and must reproduce this digest
        // before the candidate is accepted. Finding and checking are deliberately separate,
        // because a structural lookup that finds the wrong thing is not detectable by the
        // lookup itself.
        ios_seek(&k, 0);
        ios_trunc(&k, 0);
        uint32_t len = 0;
        // the digest above is the identity, taken with references off; what follows is the
        // locator, which may cite this image's own import entries
        extkey_import_index = &s->import_index;
        if (h && extkey_write(&k, (jl_value_t*)s->import_objs.items[i], 0))
            len = (uint32_t)ios_pos(&k);
        extkey_import_index = NULL;
        write_uint32(f, len);
        if (len) {
            ios_write(f, k.buf, len);
            keybytes += len;
        }
        else {
            const char *ukind = jl_typeof_str(v);
            int q;
            for (q = 0; q < nuk; q++)
                if (strcmp(uk_name[q], ukind) == 0)
                    break;
            if (q == nuk && nuk < UK_MAX) {
                uk_name[nuk] = ukind;
                uk_count[nuk] = 0;
                uk_nex[nuk] = 0;
                nuk++;
            }
            if (q < nuk) {
                uk_count[q]++;
                if (uk_nex[q] < UK_NEX)
                    uk_ex[q][uk_nex[q]++] = v;
            }
            // A free-variable type or type variable may still be anchorable without any
            // owner in this table: if it is the wrapper of its own type name, unwrapped
            // some number of times, then the type name plus that depth names it. Count how
            // many are reachable that way before building anything that depends on it.
            if (jl_is_datatype(v)) {
                jl_datatype_t *dv = (jl_datatype_t*)v;
                jl_value_t *w = dv->name->wrapper;
                int hit = (w == v);
                while (!hit && w != NULL && jl_is_unionall(w)) {
                    w = ((jl_unionall_t*)w)->body;
                    hit = (w == v);
                }
                if (hit) {
                    uk_wrap[0]++;
                }
                else {
                    // Not the wrapper itself, but its free variables may still be the
                    // wrapper's own variables -- `Array{T,1}` with `Array`'s own `T`. If
                    // so, the type name plus each variable's binding depth names them and
                    // no owner in this table is needed.
                    int all_own = 1;
                    size_t np = jl_nparams(dv);
                    for (size_t pi = 0; pi < np && all_own; pi++) {
                        jl_value_t *p = jl_tparam(dv, pi);
                        if (!jl_is_typevar(p))
                            continue;
                        int found = 0;
                        jl_value_t *ww = dv->name->wrapper;
                        while (ww != NULL && jl_is_unionall(ww)) {
                            if ((jl_value_t*)((jl_unionall_t*)ww)->var == p) { found = 1; break; }
                            ww = ((jl_unionall_t*)ww)->body;
                        }
                        all_own = found;
                    }
                    uk_wrap[all_own ? 3 : 1]++;
                }
            }
            else if (jl_is_typevar(v)) {
                uk_wrap[2]++;
            }
        }
    }
    ios_close(&k);
    if (getenv("JULIA_IMPORT_KEYS")) {
        jl_safe_printf("IMPORTKEYS_WRITE entries=%zu keybytes=%zu\n", n, keybytes);
        for (int r = 0; r < EK_CI_NREASON; r++)
            if (extkey_ci_fail[r])
                jl_safe_printf("IMPORTKEYS_CIFAIL %-16s %zu\n", extkey_ci_reason[r], extkey_ci_fail[r]);
        for (int q = 0; q < extkey_rc_n; q++)
            jl_safe_printf("IMPORTKEYS_RETCONST %-16s %zu\n", extkey_rc_name[q], extkey_rc_count[q]);
        jl_safe_printf("IMPORTKEYS_WRAPBODY body=%zu ownvars=%zu neither=%zu baretypevar=%zu\n",
                       uk_wrap[0], uk_wrap[3], uk_wrap[1], uk_wrap[2]);
        for (int q = 0; q < nuk; q++) {
            jl_safe_printf("IMPORTKEYS_UNKEYED %-20s %zu\n", uk_name[q], uk_count[q]);
            for (size_t x = 0; x < uk_nex[q]; x++) {
                // just the name: `jl_static_show` recurses without a depth limit and
                // overflows the stack on the cyclic objects that land here
                jl_value_t *e = uk_ex[q][x];
                jl_datatype_t *dt = jl_is_datatype(e) ? (jl_datatype_t*)e : NULL;
                if (jl_is_code_instance(e))
                    dt = (jl_datatype_t*)jl_typeof(e);
                jl_method_instance_t *emi = NULL;
                if (jl_is_code_instance(e))
                    emi = jl_get_ci_mi((jl_code_instance_t*)e);
                else if (jl_is_method_instance(e))
                    emi = (jl_method_instance_t*)e;
                if (emi && jl_is_method(emi->def.method))
                    jl_safe_printf("IMPORTKEYS_UNKEYED_EX %-16s %s.%s\n", uk_name[q],
                                   jl_symbol_name(emi->def.method->module->name),
                                   jl_symbol_name(emi->def.method->name));
                else if (dt && jl_is_datatype(e))
                    jl_safe_printf("IMPORTKEYS_UNKEYED_EX %-16s %s.%s\n", uk_name[q],
                                   jl_symbol_name(dt->name->module->name),
                                   jl_symbol_name(dt->name->name));
                else
                    jl_safe_printf("IMPORTKEYS_UNKEYED_EX %-16s (unnamed)\n", uk_name[q]);
            }
        }
    }
#undef UK_MAX
#undef UK_NEX
}

// Compute keys for every imported object and report coverage plus injectivity: two
// distinct objects in the same owning image must never produce the same key.
static void jl_report_import_keys(jl_serializer_state *s) JL_NOTSAFEPOINT
{
    size_t n = s->import_objs.len;
    ios_t keybuf;
    ios_mem(&keybuf, n ? n * 48 : 64);
    size_t *koff = (size_t*)malloc_s((n + 1) * sizeof(size_t));
    size_t *kbeg = (size_t*)malloc_s((n ? n : 1) * sizeof(size_t));
    size_t *kdep = (size_t*)malloc_s((n ? n : 1) * sizeof(size_t));
    size_t nkeyed = 0;
    for (size_t i = 0; i < n; i++) {
        koff[i] = (size_t)ios_pos(&keybuf);
        kdep[i] = (size_t)(uintptr_t)s->import_deps.items[i];
        // prefix with the owning image so keys only need to be unique within it
        ios_printf(&keybuf, "%zu\x1f", kdep[i]);
        kbeg[i] = (size_t)ios_pos(&keybuf);
        if (extkey_write(&keybuf, (jl_value_t*)s->import_objs.items[i], 0))
            nkeyed++;
        else
            ios_seek(&keybuf, koff[i]);   // unkeyed: leave a zero-length entry
        ios_putc('\0', &keybuf);
    }
    koff[n] = (size_t)ios_pos(&keybuf);

    // Dump the keys so two builds can be compared: a key that is not identical across
    // rebuilds of its owning image is useless, however unique it is within one build.
    // The owning-image index is deliberately omitted -- it is a per-build numbering.
    const char *dumppath = getenv("JULIA_IMPORT_KEYS_DUMP");
    if (dumppath) {
        ios_t d;
        if (ios_file(&d, dumppath, 0, 1, 1, 1) != NULL) {
            for (size_t i = 0; i < n; i++) {
                if (koff[i + 1] - koff[i] > 1) {
                    ios_puts(keybuf.buf + kbeg[i], &d);
                }
                else {
                    // unkeyed: record the type name so the unkeyed population can be
                    // compared across builds too
                    jl_value_t *o = (jl_value_t*)s->import_objs.items[i];
                    ios_printf(&d, "?%s", jl_typeof_str(o));
                }
                ios_putc('\n', &d);
            }
            ios_close(&d);
        }
    }

    // injectivity: collisions between *distinct* objects would make keys unusable
    size_t ncollide = 0, ndup = 0;
    htable_t seen;
    htable_new(&seen, 0);
    for (size_t i = 0; i < n; i++) {
        char *ki = keybuf.buf + koff[i];
        if (koff[i + 1] - koff[i] <= 1)
            continue;   // unkeyed
        // linear probe over a pointer table keyed by the string's hash
        uintptr_t h = 5381;
        for (char *c = ki; *c; c++)
            h = h * 33 + (unsigned char)*c;
        void **bp = ptrhash_bp(&seen, (void*)(h | 1));
        if (*bp != HT_NOTFOUND) {
            size_t j = (size_t)(uintptr_t)*bp - 1;
            if (strcmp(keybuf.buf + koff[j], ki) == 0 &&
                s->import_objs.items[i] != s->import_objs.items[j]) {
                // Distinct allocations that Julia itself treats as one identity are not
                // key failures: type uniquing merges equal types on load, and sibling
                // entries in a method instance's cache chain are interchangeable. Only
                // count a collision when the two are genuinely different things.
                jl_value_t *oa = (jl_value_t*)s->import_objs.items[i];
                jl_value_t *ob = (jl_value_t*)s->import_objs.items[j];
                if (extkey_equiv(oa, ob)) {
                    ndup++;
                    continue;
                }
                if (ncollide < 5) {
                    jl_value_t *a = (jl_value_t*)s->import_objs.items[i];
                    jl_value_t *b = (jl_value_t*)s->import_objs.items[j];
                    jl_safe_printf("IMPORTKEYS_COLLISION [%s vs %s] %s\n",
                                   jl_typeof_str(a), jl_typeof_str(b), keybuf.buf + kbeg[i]);
                    if (jl_is_datatype(a) && jl_is_datatype(b)) {
                        jl_datatype_t *da = (jl_datatype_t*)a, *db = (jl_datatype_t*)b;
                        jl_safe_printf("    types_equal=%d same_typename=%d hash_same=%d nparams a=%zu b=%zu concrete a=%d b=%d\n",
                            jl_types_equal(a, b), da->name == db->name, da->hash == db->hash,
                            jl_svec_len(da->parameters), jl_svec_len(db->parameters),
                            (int)da->isconcretetype, (int)db->isconcretetype);
                    }
                    if (jl_is_code_instance(a) && jl_is_code_instance(b)) {
                        jl_code_instance_t *ca = (jl_code_instance_t*)a;
                        jl_code_instance_t *cb = (jl_code_instance_t*)b;
                        jl_safe_printf("    worlds  [%zu,%zu] vs [%zu,%zu]\n",
                            jl_atomic_load_relaxed(&ca->min_world), jl_atomic_load_relaxed(&ca->max_world),
                            jl_atomic_load_relaxed(&cb->min_world), jl_atomic_load_relaxed(&cb->max_world));
                        jl_safe_printf("    exctype_same=%d rettype_const_same=%d purity_same=%d def_same=%d def_is_abioverride=%d\n",
                            ca->exctype == cb->exctype,
                            ca->rettype_const == cb->rettype_const,
                            jl_atomic_load_relaxed(&ca->ipo_purity_bits) == jl_atomic_load_relaxed(&cb->ipo_purity_bits),
                            ca->def == cb->def,
                            !jl_is_method_instance(ca->def));
                        jl_svec_t *ea = jl_atomic_load_relaxed(&ca->edges);
                        jl_svec_t *eb = jl_atomic_load_relaxed(&cb->edges);
                        int elen_same = ea && eb && jl_svec_len(ea) == jl_svec_len(eb);
                        int eelem_same = elen_same;
                        if (elen_same)
                            for (size_t q = 0; q < jl_svec_len(ea); q++) {
                                jl_value_t *sa = jl_svecref(ea, q), *sb = jl_svecref(eb, q);
                                if (sa != sb) {
                                    if (eelem_same)   // report only the first difference
                                        jl_safe_printf("    first differing edge slot %zu: %s vs %s%s\n",
                                            q, jl_typeof_str(sa), jl_typeof_str(sb),
                                            (jl_is_code_instance(sa) && jl_is_code_instance(sb) &&
                                             jl_get_ci_mi((jl_code_instance_t*)sa) == jl_get_ci_mi((jl_code_instance_t*)sb))
                                                ? "  [same MethodInstance -- sibling cache entries]" : "");
                                    eelem_same = 0;
                                }
                            }
                        jl_safe_printf("    edges_len_same=%d edges_elems_identical=%d (len a=%zu b=%zu)\n",
                            elen_same, eelem_same,
                            ea ? jl_svec_len(ea) : (size_t)0, eb ? jl_svec_len(eb) : (size_t)0);
                        jl_safe_printf("    edges_same=%d inferred_same=%d a{inferred=%d,invoke=%d} b{inferred=%d,invoke=%d} chained=%d\n",
                            jl_atomic_load_relaxed(&ca->edges) == jl_atomic_load_relaxed(&cb->edges),
                            jl_atomic_load_relaxed(&ca->inferred) == jl_atomic_load_relaxed(&cb->inferred),
                            jl_atomic_load_relaxed(&ca->inferred) != NULL,
                            jl_atomic_load_relaxed(&ca->invoke) != NULL,
                            jl_atomic_load_relaxed(&cb->inferred) != NULL,
                            jl_atomic_load_relaxed(&cb->invoke) != NULL,
                            jl_atomic_load_relaxed(&ca->next) == cb || jl_atomic_load_relaxed(&cb->next) == ca);
                    }
                }
                ncollide++;
            }
        }
        else {
            *bp = (void*)(uintptr_t)(i + 1);
        }
    }
    htable_free(&seen);

    // How many imports point into a rebuildable pkgimage rather than the sysimage
    // (blob 0)? Only the former exercise cross-rebuild key stability.
    size_t n_pkg = 0, n_pkg_keyed = 0;
    for (size_t i = 0; i < n; i++) {
        if (kdep[i] != 0) {
            n_pkg++;
            if (koff[i + 1] - koff[i] > 1)
                n_pkg_keyed++;
        }
    }
    // what is still unkeyed, by concrete type
    {
        // A fixed 24 slots silently dropped kinds once full, which hid CodeInstance
        // entirely on a large image and made the counts here untrustworthy. Report any
        // overflow rather than swallowing it.
        enum { UTY_MAX = 512 };
        void *uty[UTY_MAX]; size_t ucnt[UTY_MAX], unty = 0, udropped = 0;
        for (size_t i = 0; i < n; i++) {
            if (koff[i + 1] - koff[i] > 1)
                continue;
            void *ty = (void*)jl_typeof((jl_value_t*)s->import_objs.items[i]);
            size_t q;
            for (q = 0; q < unty; q++)
                if (uty[q] == ty) break;
            if (q == unty) {
                if (q == UTY_MAX) { udropped++; continue; }
                uty[q] = ty; ucnt[q] = 0; unty++;
            }
            ucnt[q]++;
        }
        for (size_t q = 0; q < unty; q++)
            jl_safe_printf("IMPORTKEYS_UNKEYED %-24s %zu\n",
                           jl_symbol_name(((jl_datatype_t*)uty[q])->name->name), ucnt[q]);
        if (udropped)
            jl_safe_printf("IMPORTKEYS_UNKEYED <overflow>              %zu\n", udropped);
    }
    // REVIEW instrumentation: per owning pkgimage, total vs unkeyed distinct imports.
    // Re-resolution is all-or-nothing per (dependent, dependency) pair, so what matters
    // is how many dependency images have at least one unkeyed import.
    {
        size_t maxdep = 0;
        for (size_t i = 0; i < n; i++)
            if (kdep[i] > maxdep) maxdep = kdep[i];
        size_t *dtot = (size_t*)calloc(maxdep + 1, sizeof(size_t));
        size_t *dunk = (size_t*)calloc(maxdep + 1, sizeof(size_t));
        for (size_t i = 0; i < n; i++) {
            dtot[kdep[i]]++;
            if (koff[i + 1] - koff[i] <= 1)
                dunk[kdep[i]]++;
        }
        // Attribute each blob to a package by naming the top-level module of the first
        // import that carries one, so the report is readable as dependency names rather
        // than blob indices.
        const char **dname = (const char**)calloc(maxdep + 1, sizeof(char*));
        for (size_t i = 0; i < n; i++) {
            if (dname[kdep[i]])
                continue;
            jl_value_t *o = (jl_value_t*)s->import_objs.items[i];
            jl_module_t *om = NULL;
            if (jl_is_datatype(o)) om = ((jl_datatype_t*)o)->name->module;
            else if (jl_is_typename(o)) om = ((jl_typename_t*)o)->module;
            else if (jl_is_method(o)) om = ((jl_method_t*)o)->module;
            else if (jl_is_module(o)) om = (jl_module_t*)o;
            if (om) {
                while (om->parent && om->parent != om)
                    om = om->parent;
                dname[kdep[i]] = jl_symbol_name(om->name);
            }
        }
        size_t ndeps = 0, ndeps_unk = 0;
        for (size_t d = 1; d <= maxdep; d++) {
            if (dtot[d]) {
                ndeps++;
                if (dunk[d]) ndeps_unk++;
                jl_safe_printf("IMPORTKEYS_DEP dep=%zu name=%s total=%zu unkeyed=%zu %s\n",
                               d, dname[d] ? dname[d] : "?", dtot[d], dunk[d],
                               dunk[d] ? "blocked" : "RELINKABLE");
                if (dunk[d]) {
                    // Name the kinds actually blocking *this* dependency. That is what has
                    // to be keyed next, and it is not the same as the global histogram: a
                    // kind with a huge global count may block nothing, while a single
                    // object of a rare kind can block an entire dependency.
                    void *bty[12]; size_t bcnt[12], bnty = 0;
                    for (size_t i = 0; i < n; i++) {
                        if (kdep[i] != d || koff[i + 1] - koff[i] > 1)
                            continue;
                        void *ty = (void*)jl_typeof((jl_value_t*)s->import_objs.items[i]);
                        size_t q;
                        for (q = 0; q < bnty; q++)
                            if (bty[q] == ty) break;
                        if (q == bnty) {
                            if (q == 12)
                                continue;
                            bty[q] = ty; bcnt[q] = 0; bnty++;
                        }
                        bcnt[q]++;
                    }
                    for (size_t q = 0; q < bnty; q++)
                        jl_safe_printf("IMPORTKEYS_BLOCKER %s %s %zu\n",
                                       dname[d] ? dname[d] : "?",
                                       jl_symbol_name(((jl_datatype_t*)bty[q])->name->name),
                                       bcnt[q]);
                }
            }
        }
        free(dname);
        jl_safe_printf("IMPORTKEYS_DEPS pkgimages=%zu with_unkeyed=%zu\n", ndeps, ndeps_unk);
        free(dtot);
        free(dunk);
    }
    jl_safe_printf("IMPORTKEYS distinct=%zu keyed=%zu unkeyed=%zu collisions=%zu keybytes=%zu "
                   "frompkgimage=%zu frompkgimage_keyed=%zu equivdupes=%zu\n",
                   n, nkeyed, n - nkeyed, ncollide, (size_t)ios_pos(&keybuf), n_pkg, n_pkg_keyed, ndup);
    free(koff);
    free(kbeg);
    free(kdep);
    ios_close(&keybuf);
}

// --- Shadow resolution ------------------------------------------------------------
//
// A key is a digest, so it cannot be inverted. Resolving one back to an object has to go
// through the uniquing machinery the restore path already uses: a module comes from the
// depmods array, a binding from `jl_get_module_binding`. This pass checks that this is
// enough, while we still have the answer: for every entry of the import table we know the
// real pointer, so we re-derive the object from its key operands alone and compare.
// Verification only -- nothing here feeds back into what is written.

enum { SHK_MODULE, SHK_BINDING, SHK_DATATYPE, SHK_METHODINSTANCE, SHK_OTHER, SHK_NKINDS };
static const char *shadow_kind_name[SHK_NKINDS] =
    { "Module", "Binding", "DataType", "MethodInstance", "other" };
enum { SHR_SAME, SHR_DIFF, SHR_UNRESOLVED, SHR_UNKEYED, SHR_SKIPPED, SHR_NOUTCOMES };
// why a re-derivation gave up; the innermost failure wins
enum { SHRR_NONE, SHRR_UNSUPPORTED, SHRR_DEPTH, SHRR_FREETYPEVARS, SHRR_MODULE,
       SHRR_TYPENAME, SHRR_PARAM, SHRR_CACHEMISS, SHRR_METHOD,
       SHRR_INTERSECTION, SHRR_NREASONS };
static const char *shadow_reason_name[SHRR_NREASONS] = {
    "none", "unsupported_kind", "too_deep", "free_typevars", "module_not_found",
    "typename_not_found", "operand_unresolved", "not_found_or_buildable",
    "method_not_found", "no_type_intersection" };

typedef struct {
    int passthrough;   // an operand was taken as-is instead of being re-derived
    int constructed;   // an equal object had to be built because none was findable
    int reason;
} shadow_ctx_t;

#define SHADOW_FAIL(r) do { if (c->reason == SHRR_NONE) c->reason = (r); } while (0)

#define SHADOW_MAX_MODPATH 24

// Write `m`'s root-first path as symbols into `out`; returns the number of components,
// or -1 if the path is deeper than `max`. This is the operand `extkey_module` renders.
static int shadow_modpath(jl_module_t *m, jl_sym_t **out, int max) JL_NOTSAFEPOINT
{
    int n = 0;
    for (jl_module_t *p = m; ; p = p->parent) {
        if (n == max)
            return -1;
        out[n++] = p->name;
        if (p->parent == NULL || p->parent == p)
            break;
    }
    for (int i = 0; i < n / 2; i++) {
        jl_sym_t *t = out[i];
        out[i] = out[n - 1 - i];
        out[n - 1 - i] = t;
    }
    return n;
}

// Re-derive a module from its name path alone. `mod_array` is the set of toplevel modules
// of the loaded images -- the serialization-time stand-in for the `depmods` array the
// restore path is handed, which is where a module operand is resolved from. Anchor on the
// deepest already-loaded module whose own path is a prefix of the target's, then walk down
// through constant bindings, which is how a submodule is bound in its parent.
// Find the module named by a root-first path. Split out from `shadow_resolve_module` so
// that the same descent can be driven by a path parsed out of a key, which is all a
// load-time relink has, rather than by one read off a live module.
static jl_module_t *shadow_find_module(jl_array_t *mod_array, jl_sym_t **want, int n) JL_GC_DISABLED
{
    jl_sym_t *have[SHADOW_MAX_MODPATH];
    jl_module_t *best = NULL;
    int bestk = 0;
    size_t nm = jl_array_nrows(mod_array);
    for (size_t i = 0; i < nm; i++) {
        jl_module_t *a = (jl_module_t*)jl_array_ptr_ref(mod_array, i);
        while (a != NULL) {
            int k = shadow_modpath(a, have, SHADOW_MAX_MODPATH);
            if (k > bestk && k <= n) {
                int ok = 1;
                for (int j = 0; j < k; j++) {
                    if (have[j] != want[j]) {
                        ok = 0;
                        break;
                    }
                }
                if (ok) {
                    bestk = k;
                    best = a;
                }
            }
            a = (a->parent == a || a->parent == NULL) ? NULL : a->parent;
        }
    }
    for (int j = bestk; best != NULL && j < n; j++) {
        // `_debug_only` because it will not allocate a binding partition, so the descent
        // reads without mutating; a submodule's binding is always already resolved.
        jl_binding_t *b = jl_get_module_binding(best, want[j], 0);
        jl_value_t *v = jl_get_latest_binding_value_if_resolved_and_const_debug_only(b);
        best = (v != NULL && jl_is_module(v)) ? (jl_module_t*)v : NULL;
    }
    return best;
}

static jl_module_t *shadow_resolve_module(jl_array_t *mod_array, jl_module_t *orig) JL_GC_DISABLED
{
    jl_sym_t *want[SHADOW_MAX_MODPATH];
    int n = shadow_modpath(orig, want, SHADOW_MAX_MODPATH);
    if (n < 0)
        return NULL;
    return shadow_find_module(mod_array, want, n);
}

// Re-derive a type name from its module and name. A generic function's type name (`#zero`
// for `typeof(zero)`) is bound in its module under exactly that name, so no special case
// is needed for those.
static jl_typename_t *shadow_resolve_typename(jl_array_t *mod_array, jl_typename_t *tn) JL_GC_DISABLED
{
    jl_module_t *m = shadow_resolve_module(mod_array, tn->module);
    if (m == NULL)
        return NULL;
    jl_binding_t *b = jl_get_module_binding(m, tn->name, 0);
    jl_value_t *v = jl_get_latest_binding_value_if_resolved_and_const_debug_only(b);
    if (v == NULL || !jl_is_type(v))
        return NULL;
    jl_value_t *uw = jl_unwrap_unionall(v);
    if (!jl_is_datatype(uw))
        return NULL;
    jl_typename_t *got = ((jl_datatype_t*)uw)->name;
    // the binding may be an alias for a type that belongs elsewhere
    return (got->name == tn->name && got->module == m) ? got : NULL;
}

// The re-derived counterpart of `extkey_binder_t`: the chain of `UnionAll` binders in
// scope, innermost first, pairing each original type variable with the fresh one built
// for it. Looking a variable up by its original pointer walks exactly the chain its de
// Bruijn index counts, so this is that index, resolved against the structure being walked
// rather than against a parsed key. Each frame lives on the C stack of the `UnionAll`
// resolution that pushed it, which is also what roots `newv`.
typedef struct shadow_binder {
    struct shadow_binder *outer;
    jl_tvar_t *oldv;
    jl_tvar_t *newv;
} shadow_binder_t;

static jl_value_t *shadow_resolve_datatype(jl_array_t *mod_array, jl_datatype_t *dt,
                                           shadow_binder_t *env, int depth,
                                           shadow_ctx_t *c) JL_GC_DISABLED;
static jl_value_t *shadow_resolve_type(jl_array_t *mod_array, jl_value_t *t,
                                       shadow_binder_t *env, int depth,
                                       shadow_ctx_t *c) JL_GC_DISABLED;

// Re-derive a type parameter. Only objects that live in another image need resolving at
// all: anything else is written into this image and is available directly on load.
static jl_value_t *shadow_resolve_param(jl_array_t *mod_array, jl_value_t *p,
                                        shadow_binder_t *env, int depth,
                                        shadow_ctx_t *c) JL_GC_DISABLED
{
    if (p == NULL)
        return NULL;
    if (!jl_object_in_image(p))
        return p;
    if (jl_is_type(p) || jl_is_typevar(p) || jl_is_vararg(p))
        return shadow_resolve_type(mod_array, p, env, depth, c);
    if (jl_is_module(p)) {
        jl_value_t *m = (jl_value_t*)shadow_resolve_module(mod_array, (jl_module_t*)p);
        if (m == NULL)
            SHADOW_FAIL(SHRR_MODULE);
        return m;
    }
    if (jl_is_symbol(p))
        return p;   // symbols are interned, so re-deriving one by name returns this same one
    jl_datatype_t *vt = (jl_datatype_t*)jl_typeof(p);
    jl_value_t *rt = shadow_resolve_datatype(mod_array, vt, env, depth + 1, c);
    if (rt != NULL && jl_is_datatype(rt)) {
        // `O:` -- a singleton is completely determined by its type
        if (jl_is_datatype_singleton((jl_datatype_t*)rt))
            return ((jl_datatype_t*)rt)->instance;
        // `b:` -- a boxed pointer-free immutable, the `2` in `Array{Float64,2}`, is its
        // type plus its bytes, and the key carries both. Re-box rather than reuse: the
        // bytes are content, the box is an address.
        const jl_datatype_layout_t *lo = ((jl_datatype_t*)rt)->layout;
        if (lo != NULL && lo->npointers == 0 && !lo->flags.haspadding &&
            jl_is_immutable((jl_datatype_t*)rt)) {
            c->constructed = 1;
            return jl_new_bits(rt, (const char*)p);
        }
        // `v:` -- a general immutable struct is its type plus its fields, so rebuild it
        // field by field. This is what carries a `NamedTuple`'s field-name tuple, the
        // `(:sizehint,)` in `NamedTuple{(:sizehint,), Tuple{Int}}`.
        size_t nf = jl_datatype_nfields((jl_datatype_t*)rt);
        if (lo != NULL && nf > 0 && jl_is_immutable((jl_datatype_t*)rt) &&
            !((jl_datatype_t*)rt)->name->abstract) {
            jl_value_t **fargs;
            jl_value_t *res = NULL;
            JL_GC_PUSH1(&rt);
            JL_GC_PUSHARGS(fargs, nf);
            size_t i;
            for (i = 0; i < nf; i++) {
                if (jl_field_isptr(vt, i)) {
                    jl_value_t *fv = jl_get_nth_field_noalloc(p, i);
                    // an undefined field has no content to re-derive from
                    if (fv == NULL)
                        break;
                    fargs[i] = shadow_resolve_param(mod_array, fv, env, depth + 1, c);
                }
                else {
                    // inline fields contribute their bytes, and the key only accepts
                    // those when they hold neither pointers nor padding
                    jl_value_t *ft = jl_field_type_concrete(vt, i);
                    const jl_datatype_layout_t *flo = jl_is_datatype(ft) ?
                        ((jl_datatype_t*)ft)->layout : NULL;
                    if (flo == NULL || flo->npointers != 0 || flo->flags.haspadding)
                        break;
                    jl_value_t *rft = shadow_resolve_type(mod_array, ft, env, depth + 1, c);
                    if (rft == NULL)
                        break;
                    fargs[i] = jl_new_bits(rft, (const char*)p + jl_field_offset(vt, i));
                }
                if (fargs[i] == NULL)
                    break;
            }
            if (i == nf) {
                c->constructed = 1;
                res = jl_new_structv((jl_datatype_t*)rt, fargs, (uint32_t)nf);
            }
            JL_GC_POP();
            JL_GC_POP();
            if (res != NULL)
                return res;
        }
    }
    // Anything left -- an object with an undefined or unkeyable field, a mutable object --
    // is passed through as itself. That is not a re-derivation: it uses the pointer we are
    // supposed to be re-deriving, so any type resolved this way is counted separately.
    c->passthrough = 1;
    return p;
}

// Flatten a Union in the same in-order sweep `extkey_union_parts` keys, resolving each
// component into the caller's rooted `comps` array.
static int shadow_union_parts(jl_array_t *mod_array, jl_value_t *t, shadow_binder_t *env,
                              int depth, shadow_ctx_t *c, jl_value_t **comps, size_t *i,
                              size_t n) JL_GC_DISABLED
{
    if (jl_is_uniontype(t))
        return shadow_union_parts(mod_array, ((jl_uniontype_t*)t)->a, env, depth, c, comps, i, n) &&
               shadow_union_parts(mod_array, ((jl_uniontype_t*)t)->b, env, depth, c, comps, i, n);
    if (*i >= n)
        return 0;
    jl_value_t *r = shadow_resolve_type(mod_array, t, env, depth, c);
    if (r == NULL)
        return 0;
    comps[(*i)++] = r;
    return 1;
}

// Re-derive anything type-level from the operands `extkey_type` records.
static jl_value_t *shadow_resolve_type(jl_array_t *mod_array, jl_value_t *t,
                                       shadow_binder_t *env, int depth,
                                       shadow_ctx_t *c) JL_GC_DISABLED
{
    if (depth > EXTKEY_MAX_TYPEDEPTH) {
        SHADOW_FAIL(SHRR_DEPTH);
        return NULL;
    }
    if (env == NULL && jl_has_free_typevars(t)) {
        // Nothing here binds this type's variables, so it has no identity of its own,
        // and `extkey_type` refuses it a key for the same reason.
        SHADOW_FAIL(SHRR_FREETYPEVARS);
        return NULL;
    }
    if (jl_is_typevar(t)) {
        for (shadow_binder_t *b = env; b != NULL; b = b->outer)
            if (b->oldv == (jl_tvar_t*)t)
                return (jl_value_t*)b->newv;
        SHADOW_FAIL(SHRR_FREETYPEVARS);
        return NULL;
    }
    if (t == (jl_value_t*)jl_bottom_type)
        return jl_type_union(NULL, 0);   // `U0<>`, the empty union
    if (jl_is_uniontype(t)) {
        // a Union is never cached, so it is always rebuilt; `jl_type_union` re-imposes
        // the canonical component order the key relies on
        size_t n = extkey_union_count(t), i = 0;
        jl_value_t **comps;
        JL_GC_PUSHARGS(comps, n);
        jl_value_t *res = NULL;
        if (shadow_union_parts(mod_array, t, env, depth + 1, c, comps, &i, n) && i == n) {
            c->constructed = 1;
            res = jl_type_union(comps, n);
        }
        else {
            SHADOW_FAIL(SHRR_PARAM);
        }
        JL_GC_POP();
        return res;
    }
    if (jl_is_unionall(t)) {
        jl_unionall_t *ua = (jl_unionall_t*)t;
        jl_value_t *lb = NULL, *ub = NULL, *body = NULL, *res = NULL;
        jl_tvar_t *nv = NULL;
        JL_GC_PUSH4(&lb, &ub, &body, &nv);
        lb = shadow_resolve_type(mod_array, ua->var->lb, env, depth + 1, c);
        ub = lb ? shadow_resolve_type(mod_array, ua->var->ub, env, depth + 1, c) : NULL;
        if (ub != NULL) {
            // The key does not carry the variable's name -- it identifies the variable by
            // its de Bruijn index -- so name the fresh one after that index. Names are
            // cosmetic: `jl_types_equal`, and hence `extkey_equiv`, ignores them.
            char nbuf[16];
            snprintf(nbuf, sizeof(nbuf), "#%d", depth);
            c->constructed = 1;
            nv = jl_new_typevar(jl_symbol(nbuf), lb, ub);
            shadow_binder_t b = { env, ua->var, nv };
            body = shadow_resolve_type(mod_array, ua->body, &b, depth + 1, c);
            // wrapping a bare `Vararg` in a `UnionAll` is deprecated and can throw; the
            // shapes that reach here in practice always have a type body
            if (body != NULL && !jl_is_vararg(body))
                res = jl_type_unionall(nv, body);
        }
        JL_GC_POP();
        if (res == NULL)
            SHADOW_FAIL(SHRR_PARAM);
        return res;
    }
    if (jl_is_vararg(t)) {
        jl_vararg_t *vm = (jl_vararg_t*)t;
        jl_value_t *T = NULL, *N = NULL, *res = NULL;
        JL_GC_PUSH2(&T, &N);
        int ok = 1;
        if (vm->T != NULL && (T = shadow_resolve_type(mod_array, vm->T, env, depth + 1, c)) == NULL)
            ok = 0;
        if (ok && vm->N != NULL &&
            (N = shadow_resolve_param(mod_array, vm->N, env, depth + 1, c)) == NULL)
            ok = 0;
        if (ok) {
            c->constructed = 1;
            // nothrow: a length that fails the checks yields NULL rather than an error
            res = (jl_value_t*)jl_wrap_vararg(T, N, 1, 1);
        }
        JL_GC_POP();
        if (res == NULL)
            SHADOW_FAIL(SHRR_PARAM);
        return res;
    }
    if (jl_is_datatype(t))
        return shadow_resolve_datatype(mod_array, (jl_datatype_t*)t, env, depth, c);
    SHADOW_FAIL(SHRR_UNSUPPORTED);
    return NULL;
}

static jl_value_t *shadow_resolve_datatype(jl_array_t *mod_array, jl_datatype_t *dt,
                                           shadow_binder_t *env, int depth,
                                           shadow_ctx_t *c) JL_GC_DISABLED
{
    if (depth > EXTKEY_MAX_TYPEDEPTH) {
        SHADOW_FAIL(SHRR_DEPTH);
        return NULL;
    }
    jl_typename_t *tn = shadow_resolve_typename(mod_array, dt->name);
    if (tn == NULL) {
        SHADOW_FAIL(SHRR_TYPENAME);
        return NULL;
    }
    size_t np = jl_svec_len(dt->parameters);
    jl_value_t *w = tn->wrapper;
    if (np == 0 && jl_is_datatype(w) && jl_svec_len(((jl_datatype_t*)w)->parameters) == 0)
        return w;   // a type that takes no parameters is its own type name's wrapper
    jl_svec_t *p = np ? jl_alloc_svec(np) : jl_emptysvec;
    JL_GC_PUSH1(&p);
    int ok = 1;
    for (size_t i = 0; i < np; i++) {
        jl_value_t *rp = shadow_resolve_param(mod_array, jl_svecref(dt->parameters, i),
                                              env, depth + 1, c);
        if (rp == NULL) {
            SHADOW_FAIL(SHRR_PARAM);
            ok = 0;
            break;
        }
        jl_svecset(p, i, rp);
    }
    jl_datatype_t *found = NULL;
    // A type mentioning a bound variable is not cacheable (see `cacheable` in jltypes.c),
    // so there is nothing to look up: it can only be rebuilt.
    int cacheable = !dt->hasfreetypevars;
    // `typekey_hash` reads key[0] unconditionally for `Type`, so never hand it an empty key
    if (ok && cacheable && (np > 0 || tn != jl_type_typename)) {
        // The type cache is keyed on exactly (type name, parameters), so a scratch type
        // carrying the re-derived ones is all `jl_lookup_cache_type_` needs to answer.
        jl_datatype_t *scratch = jl_new_uninitialized_datatype();
        scratch->name = tn;
        jl_gc_wb(scratch, tn);
        scratch->parameters = p;
        jl_gc_wb(scratch, p);
        found = jl_lookup_cache_type_(scratch);
        if (found == NULL && jl_is_datatype(w) &&
            jl_egal((jl_value_t*)((jl_datatype_t*)w)->parameters, (jl_value_t*)p))
            found = (jl_datatype_t*)w;   // a type name's wrapper is not kept in its cache
    }
    if (ok && found == NULL) {
        // Not every type is in the cache: a tuple type is only entered when all of its
        // parameters are concrete, which excludes every method signature, and nothing
        // mentioning a bound variable is entered at all. Such a type has to be rebuilt
        // instead of found -- legitimately, since a key names an equivalence class and
        // not an allocation.
        c->constructed = 1;
        if (tn == jl_tuple_typename)
            found = (jl_datatype_t*)jl_apply_tuple_type(p, 1);
        else if (jl_is_unionall(w))
            found = (jl_datatype_t*)jl_apply_type(w, jl_svec_data(p), np);
        if (found != NULL && !jl_is_datatype(found))
            found = NULL;
    }
    JL_GC_POP();
    if (found == NULL)
        SHADOW_FAIL(SHRR_CACHEMISS);
    return (jl_value_t*)found;
}

// Re-derive a method from its signature. The module and name are part of the key too, but
// the signature alone identifies it: a second definition with the same signature in the
// same module is a redefinition, which replaces the first rather than coexisting.
static jl_method_t *shadow_resolve_method(jl_array_t *mod_array, jl_method_t *m, int depth,
                                          shadow_ctx_t *c) JL_GC_DISABLED
{
    jl_value_t *sig = shadow_resolve_param(mod_array, m->sig, NULL, depth, c);
    if (sig == NULL)
        return NULL;
    jl_value_t *found = jl_methtable_lookup(sig, jl_atomic_load_acquire(&jl_world_counter));
    if (!jl_is_method(found)) {
        SHADOW_FAIL(SHRR_METHOD);
        return NULL;
    }
    return (jl_method_t*)found;
}

// Re-derive one imported object from its key operands. Returns NULL if the operands do
// not lead anywhere, and leaves `*kind` as the bucket to count this object under.
static jl_value_t *shadow_resolve(jl_array_t *mod_array, jl_value_t *v, int *kind,
                                  shadow_ctx_t *c) JL_GC_DISABLED
{
    if (jl_is_module(v)) {
        *kind = SHK_MODULE;
        jl_value_t *m = (jl_value_t*)shadow_resolve_module(mod_array, (jl_module_t*)v);
        if (m == NULL)
            SHADOW_FAIL(SHRR_MODULE);
        return m;
    }
    if (jl_is_binding(v)) {
        *kind = SHK_BINDING;
        jl_globalref_t *gr = ((jl_binding_t*)v)->globalref;
        if (gr == NULL) {
            SHADOW_FAIL(SHRR_UNSUPPORTED);
            return NULL;
        }
        jl_module_t *m = shadow_resolve_module(mod_array, gr->mod);
        if (m == NULL) {
            SHADOW_FAIL(SHRR_MODULE);
            return NULL;
        }
        // Same call the restore path makes, alloc and all: a binding that the key names
        // but that no longer exists is created empty rather than failing to resolve.
        return (jl_value_t*)jl_get_module_binding(m, gr->name, 1);
    }
    if (jl_is_datatype(v)) {
        *kind = SHK_DATATYPE;
        return shadow_resolve_type(mod_array, v, NULL, 0, c);
    }
    if (jl_is_method_instance(v)) {
        *kind = SHK_METHODINSTANCE;
        jl_method_instance_t *mi = (jl_method_instance_t*)v;
        if (!jl_is_method(mi->def.value)) {
            SHADOW_FAIL(SHRR_UNSUPPORTED);   // a toplevel thunk has no stable name
            return NULL;
        }
        jl_method_t *m = shadow_resolve_method(mod_array, mi->def.method, 1, c);
        if (m == NULL)
            return NULL;
        jl_value_t *spec = shadow_resolve_param(mod_array, mi->specTypes, NULL, 1, c);
        if (spec == NULL) {
            SHADOW_FAIL(SHRR_PARAM);
            return NULL;
        }
        // The static parameters are not a key operand of their own: they follow from the
        // method's signature and the specialization types, which are.
        jl_svec_t *env = jl_emptysvec;
        JL_GC_PUSH2(&spec, &env);
        jl_value_t *ti = jl_type_intersection_env(spec, m->sig, &env);
        jl_value_t *res = NULL;
        if (ti == jl_bottom_type)
            SHADOW_FAIL(SHRR_INTERSECTION);
        else
            res = (jl_value_t*)jl_specializations_get_linfo(m, spec, env);
        JL_GC_POP();
        return res;
    }
    *kind = SHK_OTHER;
    SHADOW_FAIL(SHRR_UNSUPPORTED);
    return NULL;
}

// Runs inside the JL_GC_DISABLED region of the save, like the uniquing calls it mirrors.
// The loader-side shape of the import table, declared here because the parser resolves
// `@i` against it; defined below with the probe that uses it.
typedef struct jl_import_entry_t {
    uint32_t depsidx;
    uint32_t loclen;
    uint64_t digest;
    uint64_t offset;
    char *loc;
    jl_value_t *resolved;
} jl_import_entry_t;

typedef struct jl_import_table_t {
    size_t n;
    jl_import_entry_t *e;
} jl_import_table_t;

// The import table keyed the way a reference is: by the (deps-index, offset) pair
// `add_external_linkage` encodes. Sorted once at load; see `relink_lookup`.
typedef struct jl_relink_ent_t {
    uint32_t depsidx;
    uint32_t idx;        // back into the import table, before `obj` is filled in
    uint64_t offset;
    jl_value_t *obj;     // what the entry resolved to, or NULL
} jl_relink_ent_t;

static int relink_ent_cmp(const void *a, const void *b) JL_NOTSAFEPOINT
{
    const jl_relink_ent_t *x = (const jl_relink_ent_t*)a, *y = (const jl_relink_ent_t*)b;
    if (x->depsidx != y->depsidx)
        return x->depsidx < y->depsidx ? -1 : 1;
    if (x->offset != y->offset)
        return x->offset < y->offset ? -1 : 1;
    return 0;
}

// A reference into a dependency carries (deps-index, offset), and an import-table entry
// records the same pair for the object it names -- so when that dependency's blob has
// moved, the pair is the only thing still tying the two together. Binary search over the
// sorted table, so no runtime map is built and the entry a reference belongs to is decided
// by the sort rather than by lookup order.
static inline jl_relink_ent_t *relink_lookup(jl_serializer_state *s, size_t depsidx, size_t offset) JL_NOTSAFEPOINT
{
    size_t lo = 0, hi = s->relink_nmap;
    jl_relink_ent_t *m = s->relink_map;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (m[mid].depsidx < depsidx || (m[mid].depsidx == depsidx && m[mid].offset < offset))
            lo = mid + 1;
        else if (m[mid].depsidx == depsidx && m[mid].offset == offset)
            return &m[mid];
        else
            hi = mid;
    }
    return NULL;
}

// True for exactly the dependencies whose blob moved and whose whole edge was accepted;
// NULL `relink_deps` (every load with the feature off) makes this one predictable branch.
static inline int relink_active(jl_serializer_state *s, size_t depsidx) JL_NOTSAFEPOINT
{
    return s->relink_deps != NULL && depsidx < s->relink_ndeps && s->relink_deps[depsidx];
}

static size_t relink_repointed = 0;   // references actually repointed, for the report

static inline uintptr_t relink_resolve(jl_serializer_state *s, size_t depsidx, size_t offset) JL_NOTSAFEPOINT
{
    jl_relink_ent_t *ent = relink_lookup(s, depsidx, offset);
    relink_repointed++;
    // Cannot happen: `add_external_linkage` records an import entry for every object it
    // encodes a reference to, and the edge was only marked relinkable after every one of
    // its entries resolved. Falling through to the offset arithmetic would compute an
    // address in the rebuilt blob, so fail loudly instead of silently wrongly.
    if (ent == NULL || ent->obj == NULL) {
        jl_safe_printf("relink: no resolved import entry for (%zu, %zu)\n", depsidx, offset);
        abort();
    }
    return (uintptr_t)ent->obj;
}

static void relink_free(jl_serializer_state *s, jl_import_table_t *tbl) JL_NOTSAFEPOINT
{
    if (relink_repointed)
        jl_safe_printf("RELINK_REPOINTED refs=%zu\n", relink_repointed);
    relink_repointed = 0;
    free(s->relink_map);
    s->relink_map = NULL;
    s->relink_nmap = 0;
    s->relink_deps = NULL;
    s->relink_ndeps = 0;
    if (tbl != NULL) {
        for (size_t i = 0; i < tbl->n; i++)
            free(tbl->e[i].loc);
        free(tbl->e);
        free(tbl);
    }
}

// ---- Reading a key back into the object it names ----
//
// The shadow pass above re-derives an import from the *live object*, which shows the
// runtime tables can find it. A load-time relink has no live object: it has the key text
// out of the image and nothing else. This parses that text and rebuilds the object from
// it. Whatever the parser cannot express is a defect in the key format rather than in the
// tables -- and is exactly what has to be found before any of this reaches the load path.

typedef struct {
    const char *p;
    const char *end;
    jl_array_t *mod_array;
    // resolving `@i` means resolving import entry `i` first, so the parser needs the table
    // and a memo: a locator graph is shared, not a tree, and re-deriving a shared entry
    // once per citation is what makes a naive resolver quadratic.
    jl_serializer_state *st;
    struct jl_import_table_t *tbl;   // set instead of `st` when reading a loaded table
    jl_value_t **memo;
    char *memo_state;   // 0 unvisited, 1 resolved, 2 in progress
    size_t nmemo;
    // binders in scope, outermost first; a `#i` counts from the innermost, so it indexes
    // this array from the end -- the same walk `extkey_type` does over its binder chain.
    jl_tvar_t *binders[EXTKEY_MAX_TYPEDEPTH];
    int nbind;
} keyparse_t;

#define KP_MAX_UNION 64

static int kp_lit(keyparse_t *kp, const char *s) JL_NOTSAFEPOINT
{
    size_t n = strlen(s);
    if ((size_t)(kp->end - kp->p) >= n && memcmp(kp->p, s, n) == 0) {
        kp->p += n;
        return 1;
    }
    return 0;
}

static int kp_char(keyparse_t *kp, char c) JL_NOTSAFEPOINT
{
    if (kp->p < kp->end && *kp->p == c) {
        kp->p++;
        return 1;
    }
    return 0;
}

static int kp_hexdigit(char c) JL_NOTSAFEPOINT
{
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return -1;
}

static int kp_uint(keyparse_t *kp, size_t *out) JL_NOTSAFEPOINT
{
    const char *s = kp->p;
    size_t v = 0;
    while (kp->p < kp->end && *kp->p >= '0' && *kp->p <= '9')
        v = v * 10 + (size_t)(*kp->p++ - '0');
    if (kp->p == s)
        return 0;
    *out = v;
    return 1;
}

// A name runs to the next structural delimiter. This is the weakest point of a textual
// key: a symbol containing one of those characters -- `var"a.b"` -- cannot be read back.
// Such a key is refused rather than misparsed, and the refusals are counted.
static int kp_name(keyparse_t *kp, jl_sym_t **out) JL_GC_DISABLED
{
    size_t n;
    if (!kp_uint(kp, &n) || !kp_char(kp, ':'))
        return 0;
    char buf[1024];
    if (n >= sizeof(buf) || kp->p + n > kp->end)
        return 0;
    memcpy(buf, kp->p, n);
    buf[n] = '\0';
    kp->p += n;
    *out = jl_symbol(buf);
    return 1;
}

// A root-first dotted path. For the kinds that name something *inside* a module the last
// component is that name, not a module.
static int kp_path(keyparse_t *kp, jl_sym_t **out, int max) JL_GC_DISABLED
{
    int n = 0;
    for (;;) {
        if (n == max)
            return -1;
        if (!kp_name(kp, &out[n]))
            return -1;
        n++;
        if (!kp_char(kp, '.'))
            return n;
    }
}

static jl_value_t *kp_type(keyparse_t *kp, int depth, int tdepth) JL_GC_DISABLED;

static jl_value_t *kp_param(keyparse_t *kp, int depth, int tdepth) JL_GC_DISABLED;

static jl_value_t *kp_value(keyparse_t *kp, int depth) JL_GC_DISABLED;

// why a code instance was not found, counted per candidate rejected and per lookup that
// found nothing: "which field disagrees" is the whole question for the chain walk
enum { KP_CI_OWNER, KP_CI_RETTYPE, KP_CI_EXCTYPE, KP_CI_CONST, KP_CI_PURITY, KP_CI_LIVE,
       KP_CI_EDGES, KP_CI_EMPTY, KP_CI_NOMATCH, KP_CI_AMBIG, KP_CI_NREASON };
static size_t kp_ci_miss[KP_CI_NREASON];
// set by `jl_relink_probe` under JULIA_PKGIMAGE_RELINK_VERBOSE so a liveness rejection
// can show *which* state each side saw, not only that they disagreed
static int kp_ci_verbose = 0;
static const char *kp_ci_reason[KP_CI_NREASON] = {
    "owner", "rettype", "exctype", "rettype_const", "purity", "liveness", "edges",
    "chain_empty", "no_match", "ambiguous"
};

// `@i` -- resolve import entry `i` by resolving its own locator. The recursion terminates
// because a locator cites only entries it does not itself contain, and an in-progress mark
// refuses the case where that stops being true rather than looping.
static jl_value_t *kp_ref(keyparse_t *kp, size_t idx) JL_GC_DISABLED
{
    if (idx == 0 || idx > kp->nmemo)
        return NULL;
    size_t i = idx - 1;
    if (kp->memo_state[i] == 1)
        return kp->memo[i];
    if (kp->memo_state[i] == 2)
        return NULL;
    kp->memo_state[i] = 2;
    // The cited entry's locator comes either from the table being read, which is the case
    // that matters, or is re-rendered from the live object, which is how the self-check
    // exercises the same path at save time.
    const char *loc = NULL;
    size_t loclen = 0;
    ios_t k;
    int rendered = 0;
    if (kp->tbl != NULL) {
        loc = kp->tbl->e[i].loc;
        loclen = kp->tbl->e[i].loclen;
    }
    else if (kp->st != NULL) {
        ios_mem(&k, 256);
        rendered = 1;
        extkey_import_index = &kp->st->import_index;
        int ok = extkey_write(&k, (jl_value_t*)kp->st->import_objs.items[i], 0);
        extkey_import_index = NULL;
        if (ok) {
            loc = k.buf;
            loclen = (size_t)ios_pos(&k);
        }
    }
    jl_value_t *got = NULL;
    if (loc != NULL && loclen != 0) {
        const char *savep = kp->p, *saveend = kp->end;
        int savenbind = kp->nbind;
        kp->p = loc;
        kp->end = loc + loclen;
        kp->nbind = 0;
        got = kp_value(kp, 0);
        if (got != NULL && kp->p != kp->end)
            got = NULL;   // a locator that does not consume its own text resolved nothing
        kp->p = savep;
        kp->end = saveend;
        kp->nbind = savenbind;
    }
    if (rendered)
        ios_close(&k);
    kp->memo[i] = got;
    kp->memo_state[i] = got != NULL ? 1 : 0;
    return got;
}

// Walk a binder root in the same pre-order the writer registered it in, and return the
// variable introduced at position `want`.
static jl_tvar_t *kp_binder_at(jl_value_t *t, int *ord, int want, int fuel) JL_NOTSAFEPOINT
{
    if (t == NULL || fuel <= 0)
        return NULL;
    if (jl_is_unionall(t)) {
        jl_unionall_t *ua = (jl_unionall_t*)t;
        int here = (*ord)++;
        if (here == want)
            return ua->var;
        jl_tvar_t *r = kp_binder_at(ua->var->lb, ord, want, fuel - 1);
        if (r == NULL)
            r = kp_binder_at(ua->var->ub, ord, want, fuel - 1);
        if (r == NULL)
            r = kp_binder_at(ua->body, ord, want, fuel - 1);
        return r;
    }
    if (jl_is_uniontype(t)) {
        jl_tvar_t *r = kp_binder_at(((jl_uniontype_t*)t)->a, ord, want, fuel - 1);
        return r ? r : kp_binder_at(((jl_uniontype_t*)t)->b, ord, want, fuel - 1);
    }
    if (jl_is_datatype(t)) {
        jl_svec_t *ps = ((jl_datatype_t*)t)->parameters;
        for (size_t i = 0; i < jl_svec_len(ps); i++) {
            jl_tvar_t *r = kp_binder_at(jl_svecref(ps, i), ord, want, fuel - 1);
            if (r != NULL)
                return r;
        }
    }
    return NULL;
}

// `T:mod.path.Name{n:p,...}` -- the type name is the last path component, and the
// parameters are applied to whatever that name is bound to, exactly as
// `shadow_resolve_datatype` does, so that every parameter is re-derived rather than reused.
static jl_value_t *kp_datatype(keyparse_t *kp, int depth, int tdepth) JL_GC_DISABLED
{
    jl_sym_t *path[SHADOW_MAX_MODPATH];
    int n = kp_path(kp, path, SHADOW_MAX_MODPATH);
    if (n < 2)
        return NULL;   // a type always lives in a module, so there is a name and a path
    jl_module_t *m = shadow_find_module(kp->mod_array, path, n - 1);
    if (m == NULL)
        return NULL;
    jl_binding_t *b = jl_get_module_binding(m, path[n - 1], 0);
    jl_value_t *v = jl_get_latest_binding_value_if_resolved_and_const_debug_only(b);
    if (v == NULL || !jl_is_type(v))
        return NULL;
    jl_value_t *uw = jl_unwrap_unionall(v);
    if (!jl_is_datatype(uw))
        return NULL;
    jl_typename_t *tn = ((jl_datatype_t*)uw)->name;
    // the binding may be an alias for a type belonging elsewhere, which would silently
    // resolve to the wrong thing
    if (tn->name != path[n - 1] || tn->module != m)
        return NULL;
    if (!kp_char(kp, '{'))
        return uw;
    size_t np;
    if (!kp_uint(kp, &np) || !kp_char(kp, ':') || np > KP_MAX_UNION)
        return NULL;
    if (np == 0)
        // `{0:}` -- an instantiation with no parameters, such as `Tuple{}`, which is not
        // the same type as the bare name it instantiates
        return kp_char(kp, '}') ? jl_apply_type(v, NULL, 0) : NULL;
    jl_value_t *params[KP_MAX_UNION];
    for (size_t i = 0; i < np; i++) {
        if (i && !kp_char(kp, ','))
            return NULL;
        params[i] = kp_param(kp, depth, tdepth + 1);
        if (params[i] == NULL)
            return NULL;
    }
    if (!kp_char(kp, '}'))
        return NULL;
    return jl_apply_type(v, params, np);
}

static jl_value_t *kp_type(keyparse_t *kp, int depth, int tdepth) JL_GC_DISABLED
{
    // two budgets, as the writer has: type nesting is counted separately from value
    // nesting, because a type reaches only types and never re-enters the value graph
    if (tdepth > EXTKEY_MAX_TYPEDEPTH || depth > EXTKEY_MAX_DEPTH)
        return NULL;
    if (kp_lit(kp, "TV@")) {
        size_t idx;
        if (!kp_uint(kp, &idx) || !kp_char(kp, '/'))
            return NULL;
        size_t pos;
        if (!kp_uint(kp, &pos))
            return NULL;
        jl_value_t *binder = kp_ref(kp, idx);
        if (binder == NULL)
            return NULL;
        int ord = 0;
        return (jl_value_t*)kp_binder_at(binder, &ord, (int)pos, EXTKEY_MAX_TYPEDEPTH);
    }
    if (kp_char(kp, '@')) {
        size_t idx;
        return kp_uint(kp, &idx) ? kp_ref(kp, idx) : NULL;
    }
    if (kp_char(kp, '#')) {
        size_t i;
        if (!kp_uint(kp, &i) || (int)i >= kp->nbind)
            return NULL;
        return (jl_value_t*)kp->binders[kp->nbind - 1 - (int)i];
    }
    if (kp_char(kp, 'U')) {
        size_t n;
        if (!kp_uint(kp, &n) || !kp_char(kp, '<'))
            return NULL;
        if (n == 0)
            return kp_char(kp, '>') ? (jl_value_t*)jl_bottom_type : NULL;
        if (n > KP_MAX_UNION)
            return NULL;
        jl_value_t *parts[KP_MAX_UNION];
        for (size_t i = 0; i < n; i++) {
            if (i && !kp_char(kp, ','))
                return NULL;
            parts[i] = kp_type(kp, depth, tdepth + 1);
            if (parts[i] == NULL)
                return NULL;
        }
        if (!kp_char(kp, '>'))
            return NULL;
        // Rebuild the tree verbatim -- right-nested over the written sequence, which is
        // the writer object's in-order flattening -- rather than through
        // `jl_type_union`. The canonicalizer of *today* may simplify what the writer's
        // object kept: measured on Makie, a union built by the intersection machinery
        // held two structurally identical components (distinct TypeVar allocations), and
        // `jl_type_union` collapsed them on rebuild, so the resolved type re-rendered
        // shorter than the identity the digest was taken over and the gate refused it,
        // cascading into every TypeVar the collapsed binder introduced. A verbatim
        // rebuild re-renders to exactly the written sequence, and such trees are legal:
        // the writer's own live object has this very structure.
        jl_value_t *u = parts[n - 1];
        for (size_t i = n - 1; i > 0; i--)
            u = jl_new_struct(jl_uniontype_type, parts[i - 1], u);
        return u;
    }
    if (kp_lit(kp, "A<")) {
        jl_value_t *lb = kp_type(kp, depth, tdepth + 1);
        if (lb == NULL || !kp_char(kp, ','))
            return NULL;
        jl_value_t *ub = kp_type(kp, depth, tdepth + 1);
        if (ub == NULL || !kp_char(kp, ';'))
            return NULL;
        // The name is not part of the identity -- a bound variable is named by its
        // position, which is why the key carries an index and not a name -- so any name
        // will do for the reconstruction.
        if (kp->nbind == EXTKEY_MAX_TYPEDEPTH)
            return NULL;
        jl_tvar_t *var = jl_new_typevar(jl_symbol("_"), lb, ub);
        kp->binders[kp->nbind++] = var;
        jl_value_t *body = kp_type(kp, depth, tdepth + 1);
        kp->nbind--;
        if (body == NULL || !kp_char(kp, '>'))
            return NULL;
        return jl_type_unionall(var, body);
    }
    if (kp_lit(kp, "X<")) {
        jl_value_t *T = NULL, *N = NULL;
        if (!kp_char(kp, '-')) {
            T = kp_type(kp, depth, tdepth + 1);
            if (T == NULL)
                return NULL;
        }
        if (!kp_char(kp, ','))
            return NULL;
        if (!kp_char(kp, '-')) {
            N = kp_param(kp, depth, tdepth + 1);
            if (N == NULL)
                return NULL;
        }
        if (!kp_char(kp, '>'))
            return NULL;
        return (jl_value_t*)jl_wrap_vararg(T, N, 1, 0);
    }
    if (kp_lit(kp, "T:"))
        return kp_datatype(kp, depth, tdepth);
    return NULL;
}

static jl_value_t *kp_value(keyparse_t *kp, int depth) JL_GC_DISABLED;

static jl_value_t *kp_param(keyparse_t *kp, int depth, int tdepth) JL_GC_DISABLED
{
    if (kp->p < kp->end && strchr("#UAXT@", *kp->p) != NULL)
        return kp_type(kp, depth, tdepth);
    return kp_value(kp, depth + 1);
}

static jl_value_t *kp_value(keyparse_t *kp, int depth) JL_GC_DISABLED
{
    if (depth > EXTKEY_MAX_DEPTH)
        return NULL;
    if (kp_lit(kp, "M:")) {
        jl_sym_t *path[SHADOW_MAX_MODPATH];
        int n = kp_path(kp, path, SHADOW_MAX_MODPATH);
        if (n < 1)
            return NULL;
        return (jl_value_t*)shadow_find_module(kp->mod_array, path, n);
    }
    if (kp_lit(kp, "N:") || kp_lit(kp, "B:")) {
        int binding = kp->p[-2] == 'B';
        jl_sym_t *path[SHADOW_MAX_MODPATH];
        int n = kp_path(kp, path, SHADOW_MAX_MODPATH);
        if (n < 2)
            return NULL;
        jl_module_t *m = shadow_find_module(kp->mod_array, path, n - 1);
        if (m == NULL)
            return NULL;
        jl_binding_t *b = jl_get_module_binding(m, path[n - 1], 0);
        if (binding)
            return (jl_value_t*)b;
        jl_value_t *v = jl_get_latest_binding_value_if_resolved_and_const_debug_only(b);
        if (v == NULL || !jl_is_type(v))
            return NULL;
        jl_value_t *uw = jl_unwrap_unionall(v);
        if (!jl_is_datatype(uw))
            return NULL;
        jl_typename_t *tn = ((jl_datatype_t*)uw)->name;
        return (tn->name == path[n - 1] && tn->module == m) ? (jl_value_t*)tn : NULL;
    }
    if (kp_lit(kp, "S:")) {
        jl_sym_t *s;
        return kp_name(kp, &s) ? (jl_value_t*)s : NULL;
    }
    if (kp_lit(kp, "s:")) {
        // a string is exactly its bytes, and `jl_egal` on strings is content-wise, so a
        // fresh allocation is the object -- the same re-box the `b:` case does. The hex
        // run is self-delimiting: no closing delimiter is a hex digit.
        const char *q = kp->p;
        while (q + 2 <= kp->end && kp_hexdigit(q[0]) >= 0 && kp_hexdigit(q[1]) >= 0)
            q += 2;
        size_t nb = (size_t)(q - kp->p) / 2;
        jl_value_t *str = jl_alloc_string(nb);
        char *d = jl_string_data(str);
        for (size_t i = 0; i < nb; i++) {
            d[i] = (char)((kp_hexdigit(kp->p[0]) << 4) | kp_hexdigit(kp->p[1]));
            kp->p += 2;
        }
        return str;
    }
    if (kp_lit(kp, "V:")) {
        // element-wise, exactly as written; `jl_egal` on simple vectors is content-wise
        // (`compare_svec`), so a fresh svec whose elements resolved is the object, the
        // same way a re-boxed immutable is.
        size_t n;
        if (!kp_uint(kp, &n) || !kp_char(kp, '<'))
            return NULL;
        jl_svec_t *sv = jl_alloc_svec(n);
        for (size_t i = 0; i < n; i++) {
            if (i && !kp_char(kp, ','))
                return NULL;
            if (kp_char(kp, '0'))
                continue;   // a NULL slot, written as `0`; no locator begins with a digit
            jl_value_t *e = kp_param(kp, depth + 1, 0);
            if (e == NULL)
                return NULL;
            jl_svecset(sv, i, e);
        }
        return kp_char(kp, '>') ? (jl_value_t*)sv : NULL;
    }
    if (kp_lit(kp, "F:")) {
        // The module and name locate the definition site for a reader; what actually finds
        // the method is the signature, which is what distinguishes the methods of one
        // generic function, so the lookup goes straight through the method table.
        jl_sym_t *path[SHADOW_MAX_MODPATH];
        int n = kp_path(kp, path, SHADOW_MAX_MODPATH);
        if (n < 2 || !kp_char(kp, '@'))
            return NULL;
        jl_value_t *sig = kp_type(kp, depth, 0);
        if (sig == NULL)
            return NULL;
        if (kp_lit(kp, "@F")) {
            // the body digest is part of the identity, not of the lookup: it decides
            // whether the method that was found may be used, which the caller checks by
            // recomputing this key over the result
            for (int i = 0; i < 16; i++)
                if (kp->p >= kp->end || kp_hexdigit(*kp->p++) < 0)
                    return NULL;
        }
        jl_value_t *found = jl_methtable_lookup(sig, jl_atomic_load_acquire(&jl_world_counter));
        return jl_is_method(found) ? found : NULL;
    }
    if (kp_lit(kp, "I:")) {
        jl_value_t *mv = kp_value(kp, depth + 1);
        if (mv == NULL || !jl_is_method(mv) || !kp_char(kp, '/'))
            return NULL;
        jl_value_t *spec = kp_type(kp, depth, 0);
        if (spec == NULL)
            return NULL;
        // The static parameters are not carried in the key: they follow from the method's
        // signature and the specialization types, which are.
        jl_svec_t *env = jl_emptysvec;
        jl_value_t *ti = jl_type_intersection_env(spec, ((jl_method_t*)mv)->sig, &env);
        if (ti == jl_bottom_type)
            return NULL;
        return (jl_value_t*)jl_specializations_get_linfo((jl_method_t*)mv, spec, env);
    }
    if (kp_lit(kp, "C:")) {
        // A code instance has no lookup of its own: it is found by walking the cache chain
        // of the method instance that owns it and matching on everything the key records.
        // That the key must record all of it is what the earlier commit established -- the
        // entries in one chain differ in exactly these fields and nothing else.
        jl_value_t *miv = kp_value(kp, depth + 1);
        if (miv == NULL || !jl_is_method_instance(miv) || !kp_char(kp, '/'))
            return NULL;
        jl_value_t *owner = jl_nothing;
        if (!kp_char(kp, '-')) {
            owner = kp_value(kp, depth + 1);
            if (owner == NULL)
                return NULL;
        }
        if (!kp_char(kp, '/'))
            return NULL;
        jl_value_t *rettype = kp_type(kp, depth, 0);
        if (rettype == NULL || !kp_lit(kp, "/E"))
            return NULL;
        uint64_t ehash = 0;
        for (int i = 0; i < 16; i++) {
            int d = kp->p < kp->end ? kp_hexdigit(*kp->p++) : -1;
            if (d < 0)
                return NULL;
            ehash = (ehash << 4) | (uint64_t)d;
        }
        if (!kp_char(kp, '/'))
            return NULL;
        jl_value_t *exctype = kp_type(kp, depth, 0);
        if (exctype == NULL || !kp_char(kp, '/'))
            return NULL;
        jl_value_t *rtc = NULL;
        if (!kp_char(kp, '-')) {
            rtc = kp_value(kp, depth + 1);
            if (rtc == NULL)
                return NULL;
        }
        if (!kp_lit(kp, "/P"))
            return NULL;
        uint32_t purity = 0;
        for (int i = 0; i < 8; i++) {
            int d = kp->p < kp->end ? kp_hexdigit(*kp->p++) : -1;
            if (d < 0)
                return NULL;
            purity = (purity << 4) | (uint32_t)d;
        }
        int live;
        if (kp_char(kp, 'L'))
            live = 1;
        else if (kp_char(kp, 'D'))
            live = 0;
        else
            return NULL;
        jl_method_instance_t *mi = (jl_method_instance_t*)miv;
        size_t nchain = 0, nmatch = 0;
        jl_code_instance_t *match = NULL;
        for (jl_code_instance_t *ci = jl_atomic_load_relaxed(&mi->cache); ci != NULL;
             ci = jl_atomic_load_relaxed(&ci->next)) {
            nchain++;
            if (!jl_egal(ci->owner, owner))
                { kp_ci_miss[KP_CI_OWNER]++; continue; }
            if (!jl_types_equal(ci->rettype, rettype))
                { kp_ci_miss[KP_CI_RETTYPE]++; continue; }
            if (!jl_types_equal(ci->exctype, exctype))
                { kp_ci_miss[KP_CI_EXCTYPE]++; continue; }
            if ((rtc == NULL) != (ci->rettype_const == NULL) ||
                (rtc != NULL && !jl_egal(rtc, ci->rettype_const)))
                { kp_ci_miss[KP_CI_CONST]++; continue; }
            if (jl_atomic_load_relaxed(&ci->ipo_purity_bits) != purity)
                { kp_ci_miss[KP_CI_PURITY]++; continue; }
            // Liveness is deliberately *not* a filter here, though the key records it.
            // Whether an instance has been invalidated is a property of the loading
            // session, not of the object: measured against ground truth, the same code
            // instance was dead when one image was written and live when that image was
            // loaded, and dead-now where the key said live. Filtering on it therefore
            // does not select the object the reference names, it selects whichever
            // sibling happens to be in the recorded state -- which is how a key that
            // reads `D` resolved to a sibling instead of to the instance the reference
            // means. Ignoring it makes the two siblings indistinguishable, so the
            // ambiguity check below refuses, and the digest gate (which does still carry
            // the bit) refuses a state that no longer matches. Both cost a rebuild.
            (void)live;
            uint64_t h;
            if (!extkey_edges_hash(ci, &h) || h != ehash)
                { kp_ci_miss[KP_CI_EDGES]++; continue; }
            // Do not stop at the first match. A key that two members of one chain both
            // satisfy names a set, not an object, and picking a member of that set is
            // guessing -- measured on Makie, 135 of the entries that resolved to the wrong
            // object were exactly this. Refusing is the conservative answer, and it costs
            // a rebuild rather than the wrong compiled code attached to a MethodInstance.
            nmatch++;
            match = ci;
        }
        if (nmatch == 1)
            return (jl_value_t*)match;
        if (nmatch > 1) {
            kp_ci_miss[KP_CI_AMBIG]++;
            return NULL;
        }
        kp_ci_miss[nchain == 0 ? KP_CI_EMPTY : KP_CI_NOMATCH]++;
        return NULL;
    }
    if (kp_lit(kp, "O:")) {
        // a singleton is completely determined by its type
        jl_value_t *t = kp_type(kp, depth, 0);
        if (t == NULL || !jl_is_datatype(t) || !jl_is_datatype_singleton((jl_datatype_t*)t))
            return NULL;
        return ((jl_datatype_t*)t)->instance;
    }
    if (kp_lit(kp, "b:")) {
        // a boxed pointer-free immutable: its type plus its bytes. Re-box rather than
        // reuse, as the shadow pass does -- the bytes are content, the box is an address.
        jl_value_t *t = kp_type(kp, depth, 0);
        if (t == NULL || !jl_is_datatype(t) || !kp_char(kp, '<'))
            return NULL;
        size_t sz = jl_datatype_size((jl_datatype_t*)t);
        char bytes[64];
        if (sz > sizeof(bytes))
            return NULL;
        for (size_t i = 0; i < sz; i++) {
            int hi, lo;
            if (kp->p + 2 > kp->end)
                return NULL;
            hi = kp_hexdigit(*kp->p++);
            lo = kp_hexdigit(*kp->p++);
            if (hi < 0 || lo < 0)
                return NULL;
            bytes[i] = (char)((hi << 4) | lo);
        }
        if (!kp_char(kp, '>'))
            return NULL;
        return jl_new_bits(t, bytes);
    }
    if (kp_lit(kp, "v:")) {
        // a general immutable struct: its type plus its fields. Rebuild a fresh object
        // from re-derived parts, exactly as `b:` re-boxes -- the serialized pointer is
        // an address, never content. `jl_egal` on immutables compares field-wise, so the
        // rebuilt object is the object. The field-wise writes below mirror the writer:
        // pointer fields recurse, inline fields are bytes, and the inline conditions
        // (concrete datatype, no pointers, no padding) are re-checked so that a byte run
        // is only ever interpreted the way it was written.
        jl_value_t *t = kp_type(kp, depth, 0);
        if (t == NULL || !jl_is_datatype(t) || !kp_char(kp, '{'))
            return NULL;
        jl_datatype_t *dt = (jl_datatype_t*)t;
        if (!jl_is_immutable(dt) || !dt->isconcretetype || dt->layout == NULL ||
            jl_is_layout_opaque(dt->layout) || dt->instance != NULL)
            return NULL;
        size_t nf = jl_datatype_nfields(dt);
        if (nf == 0)
            return NULL;
        // zero-initialized, so a `0` (NULL) pointer slot needs no write
        jl_value_t *v = jl_new_struct_uninit(dt);
        for (size_t i = 0; i < nf; i++) {
            if (i && !kp_char(kp, ','))
                return NULL;
            if (jl_field_isptr(dt, i)) {
                if (kp_char(kp, '0'))
                    continue;
                jl_value_t *fv = kp_param(kp, depth + 1, 0);
                if (fv == NULL)
                    return NULL;
                *(jl_value_t**)((char*)v + jl_field_offset(dt, i)) = fv;
                jl_gc_wb(v, fv);
            }
            else {
                jl_value_t *ft = jl_field_type_concrete(dt, i);
                if (!jl_is_datatype(ft))
                    return NULL;
                const jl_datatype_layout_t *flo = ((jl_datatype_t*)ft)->layout;
                if (flo == NULL || flo->npointers != 0 || flo->flags.haspadding)
                    return NULL;
                size_t fsz = jl_field_size(dt, i);
                char *fp = (char*)v + jl_field_offset(dt, i);
                for (size_t b = 0; b < fsz; b++) {
                    if (kp->p + 2 > kp->end)
                        return NULL;
                    int hi = kp_hexdigit(*kp->p++);
                    int lo = kp_hexdigit(*kp->p++);
                    if (hi < 0 || lo < 0)
                        return NULL;
                    fp[b] = (char)((hi << 4) | lo);
                }
            }
        }
        return kp_char(kp, '}') ? v : NULL;
    }
    if (kp_lit(kp, "R:")) {
        // a global reference is exactly the module and name it names; the module's
        // canonical globalref for that name is re-derived rather than reused
        jl_sym_t *path[SHADOW_MAX_MODPATH];
        int n = kp_path(kp, path, SHADOW_MAX_MODPATH);
        if (n < 2)
            return NULL;
        jl_module_t *m = shadow_find_module(kp->mod_array, path, n - 1);
        if (m == NULL)
            return NULL;
        return jl_module_globalref(m, path[n - 1]);
    }
    if (kp_lit(kp, "P:")) {
        // keyed by binding path: the object has no content identity of its own, so the
        // constant binding that holds it names it. Resolve to whatever that binding holds
        // *now* -- the rebuilt dependency's own object -- re-checking the writer's
        // registration conditions (resolved, constant, not a module). The digest gate
        // then requires the resolved object to re-render to this same path, which it can
        // only do if `extkey_build_paths` still records this binding as its deterministic
        // winner.
        jl_sym_t *path[SHADOW_MAX_MODPATH];
        int n = kp_path(kp, path, SHADOW_MAX_MODPATH);
        if (n < 2)
            return NULL;
        jl_module_t *m = shadow_find_module(kp->mod_array, path, n - 1);
        if (m == NULL)
            return NULL;
        jl_binding_t *b = jl_get_module_binding(m, path[n - 1], 0);
        jl_value_t *v = jl_get_latest_binding_value_if_resolved_and_const_debug_only(b);
        if (v == NULL || jl_is_module(v))
            return NULL;
        return v;
    }
    if (kp_lit(kp, "G:")) {
        // a GenericMemory is its type plus its contents: rebuild a fresh one from
        // re-derived parts, exactly as `V:` rebuilds a simple vector. The element
        // conditions (boxed recurse; inline bytes only without pointers or padding) are
        // re-checked so a byte run is only ever interpreted the way it was written.
        jl_value_t *t = kp_type(kp, depth, 0);
        if (t == NULL || !jl_is_datatype(t) ||
            ((jl_datatype_t*)t)->name != jl_genericmemory_typename)
            return NULL;
        const jl_datatype_layout_t *lo = ((jl_datatype_t*)t)->layout;
        // `instance` is the empty-memory singleton, created with the layout; without it
        // `jl_alloc_genericmemory` would throw rather than allocate
        if (lo == NULL || ((jl_datatype_t*)t)->instance == NULL || !kp_char(kp, '<'))
            return NULL;
        size_t len;
        if (!kp_uint(kp, &len) || !kp_char(kp, ':'))
            return NULL;
        if (len > (size_t)(kp->end - kp->p) + 1)
            return NULL;   // longer than the remaining text could possibly encode
        jl_genericmemory_t *m = jl_alloc_genericmemory(t, len);
        if (lo->flags.arrayelem_isboxed) {
            // fresh boxed memory is zero-initialized, so a `0` (NULL) slot needs no write
            for (size_t i = 0; i < len; i++) {
                if (i && !kp_char(kp, ','))
                    return NULL;
                if (kp_char(kp, '0'))
                    continue;
                jl_value_t *e = kp_param(kp, depth + 1, 0);
                if (e == NULL)
                    return NULL;
                jl_genericmemory_ptr_set(m, i, e);
            }
        }
        else if (lo->npointers == 0 && !lo->flags.haspadding) {
            size_t nb = len * lo->size;
            char *d = (char*)m->ptr;
            for (size_t b = 0; b < nb; b++) {
                if (kp->p + 2 > kp->end)
                    return NULL;
                int hi = kp_hexdigit(*kp->p++);
                int l0 = kp_hexdigit(*kp->p++);
                if (hi < 0 || l0 < 0)
                    return NULL;
                d[b] = (char)((hi << 4) | l0);
            }
        }
        else {
            return NULL;   // inline elements carrying pointers or padding
        }
        return kp_char(kp, '>') ? (jl_value_t*)m : NULL;
    }
    return kp_type(kp, depth, 0);
}

// For every import that has a key, parse the key and compare what comes back against the
// object the key was written for. This is the invertibility claim the whole scheme rests
// on, measured rather than asserted.
static void jl_check_key_parse(jl_serializer_state *s, jl_array_t *mod_array) JL_GC_DISABLED
{
    size_t n = s->import_objs.len;
    size_t keyed = 0, attempted = 0, same = 0, differs = 0, unparsed = 0, trailing = 0;
    size_t unsupported = 0, idbytes = 0, locbytes = 0, digest_mismatch = 0;
    ios_t k;
    ios_mem(&k, 512);
    jl_value_t **memo = (jl_value_t**)calloc(n ? n : 1, sizeof(jl_value_t*));
    char *memo_state = (char*)calloc(n ? n : 1, 1);
    for (size_t i = 0; i < n; i++) {
        jl_value_t *v = (jl_value_t*)s->import_objs.items[i];
        // size of the identity rendering, for comparison against the locator
        ios_seek(&k, 0);
        ios_trunc(&k, 0);
        if (extkey_write(&k, v, 0))
            idbytes += (size_t)ios_pos(&k);
        // what a load-time relink actually reads: the locator
        ios_seek(&k, 0);
        ios_trunc(&k, 0);
        extkey_import_index = &s->import_index;
        int ok = extkey_write(&k, v, 0);
        extkey_import_index = NULL;
        if (!ok)
            continue;
        locbytes += (size_t)ios_pos(&k);
        keyed++;
        size_t len = (size_t)ios_pos(&k);
        ios_putc('\0', &k);   // the key is not NUL-terminated in the stream; printing needs it
        // only the kinds this parser covers so far; the rest are counted, not guessed at
        if (len < 2 || strchr("MNBSTObvFICVsGRP#UAX@", k.buf[0]) == NULL) {
            unsupported++;
            continue;
        }
        attempted++;
        keyparse_t kp;
        kp.p = k.buf;
        kp.end = k.buf + len;
        kp.mod_array = mod_array;
        kp.st = s;
        kp.tbl = NULL;
        kp.memo = memo;
        kp.memo_state = memo_state;
        kp.nmemo = n;
        kp.nbind = 0;
        jl_value_t *got = kp_value(&kp, 0);
        if (got == NULL) {
            if (unparsed < 6)
                jl_safe_printf("KEYPARSE_FAIL [%s] at+%zu %s\n", jl_typeof_str(v),
                               (size_t)(kp.p - k.buf), k.buf);
            unparsed++;
        }
        else if (kp.p != kp.end) {
            trailing++;
        }
        else if (got == v || extkey_equiv(got, v)) {
            // The loader has no `v` to compare against -- that is the whole point of a
            // relink -- so the acceptance test it will actually run is: recompute the
            // identity of what was resolved, and require it to reproduce the digest the
            // image recorded. Check that here too, against the object comparison, so the
            // two cannot silently disagree.
            uint64_t want = 0, gotdigest = 0;
            extkey_hash(v, &want);
            if (!extkey_hash(got, &gotdigest) || want != gotdigest)
                digest_mismatch++;
            same++;
        }
        else {
            // the only outcome that would be a miscompile rather than a missed reuse
            if (differs < 6) {
                jl_safe_printf("KEYPARSE_DIFF want[%s] got[%s] key=%s\n",
                               jl_typeof_str(v), jl_typeof_str(got), k.buf);
                jl_safe_printf("    want="); jl_static_show(JL_STDERR, v);
                jl_safe_printf("\n    got ="); jl_static_show(JL_STDERR, got);
                jl_safe_printf("\n");
            }
            differs++;
        }
    }
    ios_close(&k);
    free(memo);
    free(memo_state);
    jl_safe_printf("KEYPARSE keyed=%zu covered=%zu same=%zu differs=%zu unparsed=%zu trailing=%zu out_of_scope=%zu digest_mismatch=%zu\n",
                   keyed, attempted, same, differs, unparsed, trailing, unsupported, digest_mismatch);
    for (int r = 0; r < KP_CI_NREASON; r++)
        if (kp_ci_miss[r])
            jl_safe_printf("KEYPARSE_CI %-14s %zu\n", kp_ci_reason[r], kp_ci_miss[r]);
    memset(kp_ci_miss, 0, sizeof(kp_ci_miss));
    jl_safe_printf("KEYSIZE identity=%zu locator=%zu ratio=%.1fx\n",
                   idbytes, locbytes, locbytes ? (double)idbytes / (double)locbytes : 0.0);
}

// The leading tag of a locator names the kind of object it locates; used only for
// reporting, so an unknown first byte is a category of its own rather than an error.
static const char *relink_loc_kind(const char *loc, uint32_t len) JL_NOTSAFEPOINT
{
    if (len == 0)
        return "(empty)";
    if (len >= 2 && loc[0] == 'M' && loc[1] == 'T')
        return "MT: methodtable";
    if (len >= 2 && loc[0] == 'T' && loc[1] == 'V')
        return "TV typevar";
    switch (loc[0]) {
    case 'C': return "C: codeinst";
    case 'I': return "I: methodinst";
    case 'F': return "F: method";
    case 'T': return "T: datatype";
    case 'M': return "M: module";
    case 'B': return "B: binding";
    case 'N': return "N: typename";
    case 'S': return "S: symbol";
    case 'O': return "O: singleton";
    case 'b': return "b: boxed";
    case 's': return "s: string";
    case 'v': return "v: struct";
    case 'D': return "D: debuginfo";
    case 'V': return "V: simplevec";
    case 'G': return "G: memory";
    case 'R': return "R: globalref";
    case 'P': return "P: pathkeyed";
    case '@': return "@ reference";
    case '#': return "# binder";
    case 'A': return "A< unionall";
    case 'U': return "U union";
    case 'X': return "X< vararg";
    }
    return "(other)";
}

// ---- The import table as the loader sees it ----
//
// At save time the table is a list of live objects. At load time it is this: a deps-index,
// the offset the object had in that dependency's blob, the identity digest, and the
// locator. The offset is what ties an entry back to the references that point at it,
// without any change to how a reference is encoded.
// Resolve every locator in a loaded table against the dependencies as they exist *now*,
// and accept an entry only when the object found reproduces the digest recorded for it.
// This is the load-time half of the scheme, doing the work but not yet consuming the
// answer: no reference is repointed, so a wrong answer here cannot become a wrong program.
static int relink_ci_compiled(jl_code_instance_t *ci) JL_NOTSAFEPOINT
{
    return (jl_atomic_load_relaxed(&ci->flags) & JL_CI_FLAGS_SPECPTR_SPECIALIZED) &&
           jl_atomic_load_relaxed(&ci->specptr.fptr) != NULL;
}

// An `external_fns` slot is consumed as a machine-code entry point: `jl_root_new_gvars`
// takes the CodeInstance's `specptr.fptr` and asserts it exists. A CodeInstance key names
// an *equivalence class* whose members may differ in compilation state -- right for a data
// reference, wrong here -- so pick a member that is actually compiled, and treat "the
// rebuilt dependency compiled none of them" as unresolvable rather than repointing at a
// slot with no code behind it.
static jl_value_t *relink_compiled_ci(jl_value_t *v, uint64_t digest) JL_NOTSAFEPOINT
{
    if (!jl_is_code_instance(v))
        return NULL;
    if (relink_ci_compiled((jl_code_instance_t*)v))
        return v;
    jl_method_instance_t *mi = jl_get_ci_mi((jl_code_instance_t*)v);
    jl_value_t *found = NULL;
    for (jl_code_instance_t *c = jl_atomic_load_relaxed(&mi->cache); c != NULL;
         c = jl_atomic_load_relaxed(&c->next)) {
        if (!relink_ci_compiled(c))
            continue;
        uint64_t h = 0;
        if (extkey_hash((jl_value_t*)c, &h) && h == digest) {
            if (found != NULL)
                return NULL;   // two compiled members of one class: which one is a guess
            found = (jl_value_t*)c;
        }
    }
    return found;
}

// Returns whether every dependency whose build_id moved came through with its whole edge
// accepted -- the condition for repointing this image instead of rebuilding it. Fills in
// `tbl->e[i].resolved` either way.
static int jl_relink_probe(jl_serializer_state *s, jl_import_table_t *tbl, jl_array_t *depmods, const uint8_t *extfn) JL_GC_DISABLED
{
    // scoped to this image's dependency set, like the type-variable table below
    extkey_reset_paths();
    extkey_build_paths(depmods);
    int verbose = getenv("JULIA_PKGIMAGE_RELINK_VERBOSE") != NULL;
    size_t keyed = 0, resolved = 0, accepted = 0, unresolved = 0, digest_bad = 0, extfn_bad = 0, fabricated = 0;
    jl_value_t **memo = (jl_value_t**)calloc(tbl->n ? tbl->n : 1, sizeof(jl_value_t*));
    char *state = (char*)calloc(tbl->n ? tbl->n : 1, 1);
    // failures by locator kind, with a few examples of each kept for the report
#define RK_MAX 24
#define RK_NEX 5
    const char *rk_name[RK_MAX];
    size_t rk_unres[RK_MAX], rk_mis[RK_MAX], rk_nex[RK_MAX];
    size_t rk_ex[RK_MAX][RK_NEX], rk_exat[RK_MAX][RK_NEX];
    int nrk = 0;
    // per-dependency tallies: re-linking is all-or-nothing per edge, so the number that
    // matters is how many dependencies come through with every entry accepted
    uint32_t maxdep = 0;
    for (size_t i = 0; i < tbl->n; i++)
        if (tbl->e[i].depsidx > maxdep)
            maxdep = tbl->e[i].depsidx;
    size_t *dep_entries = (size_t*)calloc(maxdep + 1, sizeof(size_t));
    size_t *dep_keyed = (size_t*)calloc(maxdep + 1, sizeof(size_t));
    size_t *dep_resolved = (size_t*)calloc(maxdep + 1, sizeof(size_t));
    size_t *dep_accepted = (size_t*)calloc(maxdep + 1, sizeof(size_t));
    jl_value_t **dep_rep = (jl_value_t**)calloc(maxdep + 1, sizeof(jl_value_t*));
    // per-dependency failure breakdown: one failing entry refuses a whole edge, so the
    // decision-relevant number is how many entries block each edge, and of what kind
    size_t *dep_mis = (size_t*)calloc(maxdep + 1, sizeof(size_t));
    size_t *dep_kind_unres = (size_t*)calloc((size_t)(maxdep + 1) * RK_MAX, sizeof(size_t));
    // An entry the *writer* could not render carries no locator, so nothing can re-derive
    // it and its edge can never be re-linked however good the parser gets. That makes the
    // unkeyed population, not the failing one, the ceiling on `fully_accepted_all` -- so
    // name what those objects are. The blob offset recorded with each entry is what
    // identifies them here: it is the same (deps-index, offset) a reference carries.
    size_t *dep_unkeyed = (size_t*)calloc(maxdep + 1, sizeof(size_t));
    memset(kp_ci_miss, 0, sizeof(kp_ci_miss));
    kp_ci_verbose = verbose;
    // Pass 1: resolve every locator. The identity checks wait for pass 2, because
    // re-rendering an object that contains a bare TypeVar needs the type-variable table
    // below, and that table is built from the resolved objects.
    jl_value_t **resolved_objs = (jl_value_t**)calloc(tbl->n ? tbl->n : 1, sizeof(jl_value_t*));
    size_t *fail_at = (size_t*)calloc(tbl->n ? tbl->n : 1, sizeof(size_t));
    for (size_t i = 0; i < tbl->n; i++) {
        if (tbl->e[i].loclen == 0)
            continue;
        keyparse_t kp;
        kp.p = kp.end = tbl->e[i].loc;
        kp.mod_array = depmods;
        kp.st = NULL;
        kp.tbl = tbl;
        kp.memo = memo;
        kp.memo_state = state;
        kp.nmemo = tbl->n;
        kp.nbind = 0;
        // Resolve through the memo, exactly as a citation of this entry would: an entry
        // must resolve to *one* object however it is reached. A second, direct parse
        // would rebuild a second copy of anything not interned -- a fresh `UnionAll`
        // binds fresh variables, so the copy registered in the type-variable table below
        // would not be the copy a `TV@` citation walks, and every such variable would
        // fail to re-render.
        jl_value_t *got = kp_ref(&kp, i + 1);
        if (got != NULL) {
            resolved_objs[i] = got;
        }
        else {
            // re-parse only to recover how far the locator got, for the failure report
            kp.p = tbl->e[i].loc;
            kp.end = kp.p + tbl->e[i].loclen;
            kp.nbind = 0;
            (void)kp_value(&kp, 0);
            fail_at[i] = (size_t)(kp.p - tbl->e[i].loc);
        }
    }
    // Rebuild the type-variable table from the *resolved* objects, in table order --
    // exactly the writer's own loop over its import list, so a bare TypeVar re-renders
    // to the same `TV:binder-digest/position` text the digest was taken over. Without
    // this the writer-side table (built only at save time) is absent here, every
    // resolved TypeVar fails to re-render, and the gate refuses it despite the
    // resolution being correct. The table is scoped to this image: entries another
    // image registered would move the deterministic winners, so reset first.
    extkey_reset_tvars();
    htable_new(&extkey_tvars, 0);
    extkey_tvars_ready = 1;
    for (size_t i = 0; i < tbl->n; i++) {
        jl_value_t *v = resolved_objs[i];
        if (v == NULL || !jl_is_type(v))
            continue;
        uint64_t h = 0;
        if (extkey_hash(v, &h))
            { int ord = 0; extkey_register_tvars(v, h, v, &ord, EXTKEY_MAX_TYPEDEPTH); }
    }
    // Pass 2: the digest gate, unchanged -- the identity of what was resolved must
    // reproduce the digest the image recorded, byte for byte.
    for (size_t i = 0; i < tbl->n; i++) {
        uint32_t d = tbl->e[i].depsidx;
        dep_entries[d]++;
        if (tbl->e[i].loclen == 0) {
            // What *kind* of object this is cannot be asked here: the recorded offset is
            // only meaningful against the layout the image was written against, and the
            // probe runs precisely when that dependency may have been rebuilt, so the
            // address can land mid-object. The writer names these instead (RELINK_UNKEYED
            // below, under JULIA_IMPORT_KEYS), where the object is live.
            dep_unkeyed[d]++;
            continue;
        }
        dep_keyed[d]++;
        keyed++;
        jl_value_t *got = resolved_objs[i];
        int failed = got == NULL;
        int mismatched = 0;
        if (!failed) {
            dep_resolved[d]++;
            resolved++;
            uint64_t h = 0;
            int ok = extkey_hash(got, &h) && h == tbl->e[i].digest;
            // Resolution must *find* the object, not rebuild an equal one. A reference
            // means the specific object the dependency owns, and the digest cannot tell
            // the two apart: `Ref{T} where T` re-renders identically whether it is Core's
            // wrapper or a fresh UnionAll binding a fresh TypeVar. Repointing at the copy
            // gives the process two types that are equal and not identical, and every
            // concrete type instantiated from the copy is a duplicate too -- a miscompile,
            // not a wrong answer. Anything the parser fabricated lives in the heap rather
            // than in a loaded image, which is exactly the distinction to gate on.
            //
            // Blob identity is the whole of that test, not merely "in some image". Every
            // reference to this entry is encoded as (depsidx, offset) and means the object
            // at that offset in *that* dependency's blob, so an object owned by any other
            // image is not what the reference names however well it re-renders. Measured
            // on Makie: 40 of the 42 resolutions that disagreed with ground truth were
            // sibling code instances found in a different image than the one the reference
            // cites, and the digest gate cannot see the difference because content is
            // exactly what the two share.
            size_t wantblob = ~(size_t)0;
            if (d < jl_array_len(s->buildid_depmods_idxs))
                wantblob = jl_array_data(s->buildid_depmods_idxs, uint32_t)[d];
            if (ok && external_blob_index(got) != wantblob) {
                ok = 0;
                fabricated++;
            }
            // an entry consumed as an `external_fns` slot needs a *compiled* member of the
            // equivalence class the key names, not merely a member of it
            if (ok && extfn != NULL && extfn[i]) {
                jl_value_t *fn = relink_compiled_ci(got, tbl->e[i].digest);
                if (fn == NULL || external_blob_index(fn) != wantblob) {
                    ok = 0;
                    extfn_bad++;
                }
                else {
                    got = fn;
                }
            }
            if (ok) {
                accepted++;
                dep_accepted[d]++;
                // the name below comes from the representative's blob, so a rebuilt
                // object (a fresh svec or re-boxed immutable lives in no blob) cannot
                // serve as one
                if (dep_rep[d] == NULL && external_blob_index(got) < n_linkage_blobs())
                    dep_rep[d] = got;
                tbl->e[i].resolved = got;
            }
            else {
                digest_bad++;
                mismatched = 1;
                // full texts survive only in a file: `jl_safe_printf` truncates, and the
                // interesting mismatches are exactly the giant nested unionalls
                const char *misdump = getenv("JULIA_PKGIMAGE_RELINK_MISDUMP");
                if (misdump) {
                    ios_t md;
                    if (ios_file(&md, misdump, 1, 1, 1, 0) != NULL) {
                        ios_seek_end(&md);
                        ios_printf(&md, "ENTRY %zu digest=%016" PRIx64 "\nLOC %s\nID ",
                                   i, tbl->e[i].digest, tbl->e[i].loc);
                        if (!extkey_write(&md, got, 0))
                            ios_puts("(unrenderable)", &md);
                        ios_putc('\n', &md);
                        ios_close(&md);
                    }
                }
                if (verbose) {
                    // the recomputed identity, so a mismatch shows *what* re-rendered
                    // differently rather than only that something did
                    ios_t idbuf;
                    ios_mem(&idbuf, 256);
                    int idok = extkey_write(&idbuf, got, 0);
                    ios_putc('\0', &idbuf);
                    jl_safe_printf("RELINK_DIGEST_MISMATCH [%s] %s\n    id=%s\n",
                                   jl_typeof_str(got), tbl->e[i].loc,
                                   idok ? idbuf.buf : "(unrenderable)");
                    ios_close(&idbuf);
                }
            }
        }
        else {
            unresolved++;
            if (verbose)
                jl_safe_printf("RELINK_UNRESOLVED at+%zu %s\n",
                               fail_at[i], tbl->e[i].loc);
        }
        if (failed || mismatched) {
            const char *kind = relink_loc_kind(tbl->e[i].loc, tbl->e[i].loclen);
            int q;
            for (q = 0; q < nrk; q++)
                if (rk_name[q] == kind)
                    break;
            if (q == nrk && nrk < RK_MAX) {
                rk_name[nrk] = kind;
                rk_unres[nrk] = rk_mis[nrk] = rk_nex[nrk] = 0;
                nrk++;
            }
            if (q < nrk) {
                if (failed) {
                    rk_unres[q]++;
                    dep_kind_unres[(size_t)d * RK_MAX + q]++;
                    if (rk_nex[q] < RK_NEX) {
                        rk_ex[q][rk_nex[q]] = i;
                        rk_exat[q][rk_nex[q]] = fail_at[i];
                        rk_nex[q]++;
                    }
                }
                else {
                    rk_mis[q]++;
                    dep_mis[d]++;
                }
            }
        }
    }
    free(memo);
    free(state);
    free(resolved_objs);
    free(fail_at);
    jl_safe_printf("RELINK_PROBE entries=%zu keyed=%zu resolved=%zu accepted=%zu unresolved=%zu digest_mismatch=%zu uncompiled_extfn=%zu fabricated=%zu\n",
                   tbl->n, keyed, resolved, accepted, unresolved, digest_bad, extfn_bad, fabricated);
    // kinds sorted by unresolved count, mismatches alongside; examples for the top two
    int order[RK_MAX];
    for (int q = 0; q < nrk; q++)
        order[q] = q;
    for (int a = 1; a < nrk; a++) {
        int t = order[a], b = a;
        while (b > 0 && rk_unres[order[b - 1]] < rk_unres[t]) {
            order[b] = order[b - 1];
            b--;
        }
        order[b] = t;
    }
    for (int q = 0; q < nrk; q++)
        jl_safe_printf("RELINK_KIND %-16s unresolved=%zu digest_mismatch=%zu\n",
                       rk_name[order[q]], rk_unres[order[q]], rk_mis[order[q]]);
    for (int q = 0; q < nrk && q < 2; q++) {
        int k = order[q];
        for (size_t x = 0; x < rk_nex[k]; x++) {
            size_t i = rk_ex[k][x];
            jl_safe_printf("RELINK_EXAMPLE %-16s at+%zu %.200s\n",
                           rk_name[k], rk_exat[k][x], tbl->e[i].loc);
        }
    }
    for (int r = 0; r < KP_CI_NREASON; r++)
        if (kp_ci_miss[r])
            jl_safe_printf("RELINK_CI %-14s %zu\n", kp_ci_reason[r], kp_ci_miss[r]);
    memset(kp_ci_miss, 0, sizeof(kp_ci_miss));
    // per-dependency: an edge survives a rebuild only if every entry from it is accepted,
    // keyed and unkeyed alike -- an unkeyed entry cannot be re-linked at all
    size_t ndeps = 0, full_keyed = 0, full_all = 0;
    for (uint32_t d = 0; d <= maxdep; d++) {
        if (dep_entries[d] == 0)
            continue;
        ndeps++;
        if (dep_accepted[d] == dep_keyed[d])
            full_keyed++;
        if (dep_accepted[d] == dep_entries[d])
            full_all++;
        // best-effort name: the blob an accepted object lives in belongs to one dependency
        const char *name = "?";
        if (dep_rep[d] != NULL) {
            size_t blob = external_blob_index(dep_rep[d]);
            for (size_t mi = 0; mi < jl_array_nrows(depmods); mi++) {
                jl_value_t *m = jl_array_ptr_ref(depmods, mi);
                if (jl_is_module(m) && external_blob_index(m) == blob) {
                    name = jl_symbol_name(((jl_module_t*)m)->name);
                    break;
                }
            }
        }
        jl_safe_printf("RELINK_DEP idx=%u name=%s entries=%zu keyed=%zu resolved=%zu accepted=%zu\n",
                       d, name, dep_entries[d], dep_keyed[d], dep_resolved[d], dep_accepted[d]);
        if (dep_unkeyed[d])
            jl_safe_printf("RELINK_UNKEYED_DEP idx=%u name=%s entries=%zu unkeyed=%zu\n",
                           d, name, dep_entries[d], dep_unkeyed[d]);
        if (dep_accepted[d] != dep_keyed[d]) {
            // the edge is refused; name exactly what blocks it, by locator kind
            jl_safe_printf("RELINK_BLOCKED idx=%u name=%s keyed=%zu failed=%zu:",
                           d, name, dep_keyed[d], dep_keyed[d] - dep_accepted[d]);
            for (int q = 0; q < nrk; q++)
                if (dep_kind_unres[(size_t)d * RK_MAX + q])
                    jl_safe_printf(" [%s]=%zu", rk_name[q], dep_kind_unres[(size_t)d * RK_MAX + q]);
            if (dep_mis[d])
                jl_safe_printf(" [digest_mismatch]=%zu", dep_mis[d]);
            jl_safe_printf("\n");
        }
    }
    jl_safe_printf("RELINK_DEPS ndeps=%zu fully_accepted_keyed=%zu (%.1f%%) fully_accepted_all=%zu (%.1f%%)\n",
                   ndeps, full_keyed, ndeps ? 100.0 * (double)full_keyed / (double)ndeps : 0.0,
                   full_all, ndeps ? 100.0 * (double)full_all / (double)ndeps : 0.0);
    // Only the dependencies that actually moved have to be repointed, and repointing is
    // all-or-nothing per edge: one entry of a rebuilt dependency that did not resolve
    // means this image cannot be repointed at all and the caller falls back to a rebuild.
    // A moved dependency nothing imports from costs nothing.
    int relinkable = 1;
    size_t moved = 0, moved_blocked = 0;
    for (size_t d = 0; d < relink_mismatched_ndeps; d++) {
        if (!relink_mismatched_deps[d] || d > maxdep || dep_entries[d] == 0)
            continue;
        moved++;
        if (dep_accepted[d] != dep_entries[d]) {
            moved_blocked++;
            relinkable = 0;
        }
    }
    if (relink_mismatched_ndeps)
        jl_safe_printf("RELINK_REPOINT rebuilt_deps_imported=%zu blocked=%zu -> %s\n",
                       moved, moved_blocked, relinkable ? "repoint" : "rebuild");
    // Ground truth, available whenever a dependency was *not* rebuilt: the object at
    // `blob_base + offset` is by definition the one every reference to this entry means.
    // Resolution has to reproduce it exactly, and the digest gate is what decides whether
    // it did -- so comparing the two says whether the gate is strong enough. It costs
    // nothing to run and needs no rebuild, which is what makes it a usable permanent gate.
    if (getenv("JULIA_PKGIMAGE_RELINK_SELFCHECK")) {
        size_t same = 0, differ = 0, shown = 0;
        for (size_t i = 0; i < tbl->n; i++) {
            jl_value_t *got = tbl->e[i].resolved;
            uint32_t d = tbl->e[i].depsidx;
            if (got == NULL)
                continue;
            if (d < relink_mismatched_ndeps && relink_mismatched_deps[d])
                continue;   // that blob moved; the offset no longer names anything
            if (d >= jl_array_len(s->buildid_depmods_idxs))
                continue;
            size_t bi = jl_array_data(s->buildid_depmods_idxs, uint32_t)[d];
            if (2 * bi >= jl_linkage_blobs.len)
                continue;
            jl_value_t *want = (jl_value_t*)((uintptr_t)jl_linkage_blobs.items[2 * bi] +
                                             tbl->e[i].offset * SYS_EXTERNAL_LINK_UNIT);
            if (want == got) {
                same++;
                continue;
            }
            differ++;
            if (shown < 12) {
                shown++;
                jl_safe_printf("RELINK_WRONG [%s] want=[%s]@%p got=[%s]@%p",
                               relink_loc_kind(tbl->e[i].loc, tbl->e[i].loclen),
                               jl_typeof_str(want), (void*)want, jl_typeof_str(got), (void*)got);
                // only types are printed: everything else can be an arbitrarily deep
                // graph, and a diagnostic that overflows the stack takes the load with it
                if (jl_is_type(want)) {
                    jl_safe_printf("\n    want=");
                    jl_static_show(JL_STDERR, want);
                    jl_safe_printf("\n    got =");
                    jl_static_show(JL_STDERR, got);
                }
                jl_safe_printf("\n    loc=%.300s\n", tbl->e[i].loc);
            }
        }
        jl_safe_printf("RELINK_SELFCHECK identical=%zu different=%zu\n", same, differ);
    }
    free(dep_entries);
    free(dep_keyed);
    free(dep_resolved);
    free(dep_accepted);
    free(dep_rep);
    free(dep_mis);
    free(dep_kind_unres);
    free(dep_unkeyed);
    kp_ci_verbose = 0;
    return relinkable;
#undef RK_MAX
#undef RK_NEX
}

static void jl_shadow_resolve_imports(jl_serializer_state *s, jl_array_t *mod_array) JL_GC_DISABLED
{
    size_t n = s->import_objs.len;
    size_t tally[SHK_NKINDS][SHR_NOUTCOMES];
    size_t passtally[SHK_NKINDS], builttally[SHK_NKINDS], identical[SHK_NKINDS];
    memset(tally, 0, sizeof(tally));
    memset(passtally, 0, sizeof(passtally));
    memset(builttally, 0, sizeof(builttally));
    memset(identical, 0, sizeof(identical));
    size_t reasons[SHRR_NREASONS];
    memset(reasons, 0, sizeof(reasons));
    size_t nshown = 0;
    for (size_t i = 0; i < n; i++) {
        jl_value_t *v = (jl_value_t*)s->import_objs.items[i];
        int kind = SHK_OTHER;
        shadow_ctx_t c = { 0, 0, SHRR_NONE };
        uint64_t h;
        if (!extkey_hash(v, &h)) {
            // no key at all, so there is nothing to resolve from; bucket it by kind so the
            // count lines up with the unkeyed population `jl_report_import_keys` reports
            if (jl_is_module(v)) kind = SHK_MODULE;
            else if (jl_is_binding(v)) kind = SHK_BINDING;
            else if (jl_is_datatype(v)) kind = SHK_DATATYPE;
            else if (jl_is_method_instance(v)) kind = SHK_METHODINSTANCE;
            tally[kind][SHR_UNKEYED]++;
            continue;
        }
        jl_value_t *got = shadow_resolve(mod_array, v, &kind, &c);
        if (c.passthrough)
            passtally[kind]++;
        if (c.constructed)
            builttally[kind]++;
        int outcome;
        if (kind == SHK_OTHER)
            outcome = SHR_SKIPPED;
        else if (got == NULL)
            outcome = SHR_UNRESOLVED;
        else
            outcome = extkey_equiv(v, got) ? SHR_SAME : SHR_DIFF;
        tally[kind][outcome]++;
        if (v == got)
            identical[kind]++;   // the strict subset of resolved_same: the very same object
        if (outcome == SHR_UNRESOLVED)
            reasons[c.reason]++;
        if ((outcome == SHR_DIFF || outcome == SHR_UNRESOLVED) && nshown < 10) {
            nshown++;
            jl_safe_printf("SHADOW_MISS %s %s (%s) ", shadow_kind_name[kind],
                           outcome == SHR_DIFF ? "different" : "unresolved",
                           shadow_reason_name[c.reason]);
            jl_static_show(JL_STDERR, v);
            jl_safe_printf("\n");
        }
    }
    for (int r = 1; r < SHRR_NREASONS; r++) {
        if (reasons[r])
            jl_safe_printf("SHADOW_UNRESOLVED_REASON %-22s %zu\n", shadow_reason_name[r], reasons[r]);
    }
    for (int k = 0; k < SHK_NKINDS; k++) {
        jl_safe_printf("SHADOW kind=%-16s resolved_same=%zu (identical=%zu) resolved_different=%zu "
                       "unresolved=%zu unkeyed=%zu skipped=%zu passthrough=%zu built=%zu\n",
                       shadow_kind_name[k],
                       tally[k][SHR_SAME], identical[k], tally[k][SHR_DIFF],
                       tally[k][SHR_UNRESOLVED], tally[k][SHR_UNKEYED], tally[k][SHR_SKIPPED],
                       passtally[k], builttally[k]);
    }
    size_t tot[SHR_NOUTCOMES] = {0};
    for (int k = 0; k < SHK_NKINDS; k++)
        for (int o = 0; o < SHR_NOUTCOMES; o++)
            tot[o] += tally[k][o];
    jl_safe_printf("SHADOW total=%zu resolved_same=%zu resolved_different=%zu "
                   "unresolved=%zu unkeyed=%zu skipped=%zu\n", n, tot[SHR_SAME], tot[SHR_DIFF],
                   tot[SHR_UNRESOLVED], tot[SHR_UNKEYED], tot[SHR_SKIPPED]);
}

// Return the integer `id` for `v`. Generically this is looked up in `serialization_order`,
// but symbols, small integers, and a couple of special items (`nothing` and the root Task)
// have special handling.
#define backref_id(s, v, link_ids) _backref_id(s, (jl_value_t*)(v), link_ids)
static uintptr_t _backref_id(jl_serializer_state *s, jl_value_t *v, jl_array_t *link_ids) JL_GC_DISABLED
{
    assert(v != NULL && "cannot get backref to NULL object");
    if (jl_is_symbol(v)) {
        void **pidx = ptrhash_bp(&symbol_table, v);
        void *idx = *pidx;
        if (idx == HT_NOTFOUND) {
            size_t l = strlen(jl_symbol_name((jl_sym_t*)v));
            write_uint32(s->symbols, l);
            ios_write(s->symbols, jl_symbol_name((jl_sym_t*)v), l + 1);
            size_t offset = ++nsym_tag;
            assert(offset < ((uintptr_t)1 << RELOC_TAG_OFFSET) && "too many symbols");
            idx = to_seroder_entry(offset - 1);
            *pidx = idx;
        }
        return ((uintptr_t)SymbolRef << RELOC_TAG_OFFSET) + from_seroder_entry(idx);
    }
    else if (v == (jl_value_t*)s->ptls->root_task) {
        return (uintptr_t)TagRef << RELOC_TAG_OFFSET;
    }
    else if (v == jl_nothing) {
        return ((uintptr_t)TagRef << RELOC_TAG_OFFSET) + 1;
    }
    else if (jl_typetagis(v, jl_int64_tag << 4)) {
        int64_t i64 = *(int64_t*)v + NBOX_C / 2;
        if ((uint64_t)i64 < NBOX_C)
            return ((uintptr_t)TagRef << RELOC_TAG_OFFSET) + i64 + 2;
    }
    else if (jl_typetagis(v, jl_int32_tag << 4)) {
        int32_t i32 = *(int32_t*)v + NBOX_C / 2;
        if ((uint32_t)i32 < NBOX_C)
            return ((uintptr_t)TagRef << RELOC_TAG_OFFSET) + i32 + 2 + NBOX_C;
    }
    else if (jl_typetagis(v, jl_uint8_tag << 4)) {
        uint8_t u8 = *(uint8_t*)v;
        return ((uintptr_t)TagRef << RELOC_TAG_OFFSET) + u8 + 2 + NBOX_C + NBOX_C;
    }
    if (s->incremental && jl_object_in_image(v) && !jl_copy_instead_of_import(v)) {
        assert(link_ids);
        uintptr_t item = add_external_linkage(s, v, link_ids);
        assert(item && "no external linkage identified");
        return item;
    }
    void *idx = ptrhash_get(&serialization_order, v);
    if (idx == HT_NOTFOUND) {
        jl_(jl_typeof(v));
        jl_(v);
    }
    assert(idx != HT_NOTFOUND && "object missed during jl_queue_for_serialization pass");
    assert(idx != (void*)(uintptr_t)-1 && "object missed during jl_insert_into_serialization_queue pass");
    assert(idx != (void*)(uintptr_t)-2 && "object missed during jl_insert_into_serialization_queue pass");
    return ((uintptr_t)DataRef << RELOC_TAG_OFFSET) + from_seroder_entry(idx);
}


static void record_uniquing(jl_serializer_state *s, jl_value_t *fld, uintptr_t offset) JL_NOTSAFEPOINT
{
    if (s->incremental && jl_needs_serialization(s, fld) && needs_uniquing(fld, s->query_cache)) {
        if (jl_is_datatype(fld) || jl_is_datatype_singleton((jl_datatype_t*)jl_typeof(fld)))
            arraylist_push(&s->uniquing_types, (void*)(uintptr_t)offset);
        else if (jl_is_method_instance(fld) || jl_is_binding(fld))
            arraylist_push(&s->uniquing_objs, (void*)(uintptr_t)offset);
        else
            assert(0 && "unknown object type with needs_uniquing set");
    }
}

// Save blank space in stream `s` for a pointer `fld`, storing both location and target
// in `relocs_list`.
static void write_pointerfield(jl_serializer_state *s, jl_value_t *fld) JL_NOTSAFEPOINT
{
    if (fld != NULL) {
        arraylist_push(&s->relocs_list, (void*)(uintptr_t)ios_pos(s->s));
        arraylist_push(&s->relocs_list, (void*)backref_id(s, fld, s->link_ids_relocs));
        record_uniquing(s, fld, ios_pos(s->s));
    }
    write_pointer(s->s);
}

// Save blank space in stream `s` for a pointer `fld`, storing both location and target
// in `gctags_list`.
static void write_gctaggedfield(jl_serializer_state *s, jl_datatype_t *ref) JL_NOTSAFEPOINT
{
    // jl_printf(JL_STDOUT, "gctaggedfield: position %p, value 0x%lx\n", (void*)(uintptr_t)ios_pos(s->s), ref);
    arraylist_push(&s->gctags_list, (void*)(uintptr_t)ios_pos(s->s));
    arraylist_push(&s->gctags_list, (void*)backref_id(s, ref, s->link_ids_gctags));
    write_pointer(s->s);
}


// Special handling from `jl_write_values` for modules
static void jl_write_module(jl_serializer_state *s, uintptr_t item, jl_module_t *m) JL_GC_DISABLED
{
    size_t reloc_offset = ios_pos(s->s);
    size_t tot = sizeof(jl_module_t);
    ios_write(s->s, (char*)m, tot);     // raw memory dump of the `jl_module_t` structure
    // will need to recreate the binding table for this
    arraylist_push(&s->fixup_objs, (void*)reloc_offset);

    // Handle the fields requiring special attention
    jl_module_t *newm = (jl_module_t*)&s->s->buf[reloc_offset];
    newm->name = NULL;
    arraylist_push(&s->relocs_list, (void*)(reloc_offset + offsetof(jl_module_t, name)));
    arraylist_push(&s->relocs_list, (void*)backref_id(s, m->name, s->link_ids_relocs));
    newm->parent = NULL;
    arraylist_push(&s->relocs_list, (void*)(reloc_offset + offsetof(jl_module_t, parent)));
    arraylist_push(&s->relocs_list, (void*)backref_id(s, m->parent, s->link_ids_relocs));
    jl_atomic_store_relaxed(&newm->bindings, NULL);
    arraylist_push(&s->relocs_list, (void*)(reloc_offset + offsetof(jl_module_t, bindings)));
    arraylist_push(&s->relocs_list, (void*)backref_id(s, jl_atomic_load_relaxed(&m->bindings), s->link_ids_relocs));
    jl_atomic_store_relaxed(&newm->bindingkeyset, NULL);
    arraylist_push(&s->relocs_list, (void*)(reloc_offset + offsetof(jl_module_t, bindingkeyset)));
    arraylist_push(&s->relocs_list, (void*)backref_id(s, jl_atomic_load_relaxed(&m->bindingkeyset), s->link_ids_relocs));
    newm->file = NULL;
    arraylist_push(&s->relocs_list, (void*)(reloc_offset + offsetof(jl_module_t, file)));
    arraylist_push(&s->relocs_list, (void*)backref_id(s, jl_options.strip_metadata ? jl_empty_sym : m->file , s->link_ids_relocs));
    if (jl_options.strip_metadata)
        newm->line = 0;
    newm->usings_backedges = NULL;
    arraylist_push(&s->relocs_list, (void*)(reloc_offset + offsetof(jl_module_t, usings_backedges)));
    arraylist_push(&s->relocs_list, (void*)backref_id(s, get_replaceable_field(&m->usings_backedges, 1), s->link_ids_relocs));
    newm->scanned_methods = NULL;
    arraylist_push(&s->relocs_list, (void*)(reloc_offset + offsetof(jl_module_t, scanned_methods)));
    arraylist_push(&s->relocs_list, (void*)backref_id(s, get_replaceable_field(&m->scanned_methods, 1), s->link_ids_relocs));

    // After reload, everything that has happened in this process happened semantically at
    // (for .incremental) or before jl_require_world, so reset this flag.
    jl_atomic_store_relaxed(&newm->export_set_changed_since_require_world, 0);

    // write out the usings list
    memset(&newm->usings._space, 0, sizeof(newm->usings._space));
    if (m->usings.items == &m->usings._space[0]) {
        newm->usings.items = &newm->usings._space[0];
        // Push these relocations here, to keep them in order. This pairs with the `newm->usings.items = ` below.
        arraylist_push(&s->relocs_list, (void*)(reloc_offset + offsetof(jl_module_t, usings.items)));
        arraylist_push(&s->relocs_list, (void*)(((uintptr_t)DataRef << RELOC_TAG_OFFSET) + item));
        size_t i;
        for (i = 0; i < module_usings_length(m); i++) {
            struct _jl_module_using *newm_data = module_usings_getidx(newm, i);
            struct _jl_module_using *data = module_usings_getidx(m, i);
            // TODO: Remove dead entries
            newm_data->min_world = data->min_world;
            newm_data->max_world = data->max_world;
            if (s->incremental) {
                if (data->max_world != ~(size_t)0)
                    newm_data->max_world = 0;
                newm_data->min_world = jl_require_world;
            }
            arraylist_push(&s->relocs_list, (void*)(reloc_offset + offsetof(jl_module_t, usings._space[3*i])));
            arraylist_push(&s->relocs_list, (void*)backref_id(s, data->mod, s->link_ids_relocs));
        }
        newm->usings.items = (void**)offsetof(jl_module_t, usings._space);
    }
    else {
        newm->usings.items = (void**)tot;
        arraylist_push(&s->relocs_list, (void*)(reloc_offset + offsetof(jl_module_t, usings.items)));
        arraylist_push(&s->relocs_list, (void*)(((uintptr_t)DataRef << RELOC_TAG_OFFSET) + item));
        newm = NULL; // `write_*(s->s)` below may invalidate `newm`, so defensively set it to NULL
        size_t i;
        for (i = 0; i < module_usings_length(m); i++) {
            struct _jl_module_using *data = module_usings_getidx(m, i);
            write_pointerfield(s, (jl_value_t*)data->mod);
            if (s->incremental) {
                // TODO: Drop dead ones entirely?
                write_uint(s->s, jl_require_world);
                write_uint(s->s, data->max_world == ~(size_t)0 ? ~(size_t)0 : 1);
            } else {
                write_uint(s->s, data->min_world);
                write_uint(s->s, data->max_world);
            }
            static_assert(sizeof(struct _jl_module_using) == 3*sizeof(void*), "_jl_module_using mismatch");
            tot += sizeof(struct _jl_module_using);
        }
        for (; i < module_usings_max(m); i++) {
            write_pointer(s->s);
            write_uint(s->s, 0);
            write_uint(s->s, 0);
            tot += sizeof(struct _jl_module_using);
        }
    }
    assert(ios_pos(s->s) - reloc_offset == tot);
}

static void record_memoryref(jl_serializer_state *s, size_t reloc_offset, jl_genericmemoryref_t ref) {
    ios_t *f = s->s;
    // make some header modifications in-place
    jl_genericmemoryref_t *newref = (jl_genericmemoryref_t*)&f->buf[reloc_offset];
    const jl_datatype_layout_t *layout = ((jl_datatype_t*)jl_typetagof(ref.mem))->layout;
    if (!layout->flags.arrayelem_isunion && layout->size != 0) {
        newref->ptr_or_offset = (void*)((char*)ref.ptr_or_offset - (char*)ref.mem->ptr); // relocation offset (bytes)
        arraylist_push(&s->memref_list, (void*)reloc_offset); // relocation location
        arraylist_push(&s->memref_list, NULL); // relocation target (ignored)
    }
}

static void record_memoryrefs_inside(jl_serializer_state *s, jl_datatype_t *t, size_t reloc_offset, const char *data)
{
    assert(jl_is_datatype(t));
    size_t i, nf = jl_datatype_nfields(t);
    for (i = 0; i < nf; i++) {
        size_t offset = jl_field_offset(t, i);
        if (jl_field_isptr(t, i))
            continue;
        jl_value_t *ft = jl_field_type_concrete(t, i);
        if (jl_is_uniontype(ft))
            continue;
        if (jl_is_genericmemoryref_type(ft))
            record_memoryref(s, reloc_offset + offset, *(jl_genericmemoryref_t*)(data + offset));
        else
            record_memoryrefs_inside(s, (jl_datatype_t*)ft, reloc_offset + offset, data + offset);
    }
}

static void record_gvars(jl_serializer_state *s, arraylist_t *globals) JL_GC_DISABLED
{
    for (size_t i = 0; i < globals->len; i++)
        jl_queue_for_serialization(s, globals->items[i]);
}

static void record_external_fns(jl_serializer_state *s, arraylist_t *external_fns) JL_NOTSAFEPOINT
{
    if (!s->incremental) {
        assert(external_fns->len == 0);
        (void) external_fns;
        return;
    }

    // We could call jl_queue_for_serialization here, but that should
    // always be a no-op.
#ifndef JL_NDEBUG
    for (size_t i = 0; i < external_fns->len; i++) {
        jl_code_instance_t *ci = (jl_code_instance_t*)external_fns->items[i];
        assert(jl_atomic_load_relaxed(&ci->flags) & JL_CI_FLAGS_FROM_IMAGE);
    }
#endif
}

jl_value_t *jl_find_ptr = NULL;
// The main function for serializing all the items queued in `serialization_order`
// (They are also stored in `serialization_queue` which is order-preserving, unlike the hash table used
//  for `serialization_order`).
static void jl_write_values(jl_serializer_state *s) JL_GC_DISABLED
{
    size_t l = serialization_queue.len;

    arraylist_new(&layout_table, 0);
    arraylist_grow(&layout_table, l * 2);
    memset(layout_table.items, 0, l * 2 * sizeof(void*));

    // Serialize all entries
    for (size_t item = 0; item < l; item++) {
        jl_value_t *v = (jl_value_t*)serialization_queue.items[item];           // the object
        JL_GC_PROMISE_ROOTED(v);
        assert(!(s->incremental && jl_object_in_image(v)) || jl_copy_instead_of_import(v));
        jl_datatype_t *t = (jl_datatype_t*)jl_typeof(v);
        assert((!jl_is_datatype_singleton(t) || t->instance == v) && "detected singleton construction corruption");
        int mutabl = t->name->mutabl;
        ios_t *f = s->s;
        if (t->smalltag) {
            if (t->layout->npointers == 0 || t == jl_string_type) {
                if (jl_datatype_nfields(t) == 0 || mutabl == 0 || t == jl_string_type) {
                    f = s->const_data;
                }
            }
        }

        // realign stream to expected gc alignment (16 bytes) after tag
        uintptr_t skip_header_pos = ios_pos(f) + sizeof(jl_taggedvalue_t);
        uintptr_t object_id_expected = mutabl &&
                 t != jl_datatype_type &&
                 t != jl_typename_type &&
                 t != jl_string_type &&
                 t != jl_simplevector_type &&
                 t != jl_module_type;
        if (object_id_expected)
            skip_header_pos += sizeof(size_t);
        write_padding(f, LLT_ALIGN(skip_header_pos, 16) - skip_header_pos);

        // write header
        if (object_id_expected)
            write_uint(f, jl_object_id(v));
        if (s->incremental && jl_needs_serialization(s, (jl_value_t*)t) && needs_uniquing((jl_value_t*)t, s->query_cache))
            arraylist_push(&s->uniquing_types, (void*)(uintptr_t)(ios_pos(f)|1));
        if (f == s->const_data)
            write_uint(s->const_data, ((uintptr_t)t->smalltag << 4) | GC_OLD_MARKED | GC_IN_IMAGE);
        else
            write_gctaggedfield(s, t);
        size_t reloc_offset = ios_pos(f);
        assert(item < layout_table.len && layout_table.items[item] == NULL);
        layout_table.items[item] = (void*)(reloc_offset | (f == s->const_data)); // store the inverse mapping of `serialization_order` (`id` => object-as-streampos)

        if (s->incremental) {
            if (needs_uniquing(v, s->query_cache)) {
                if (jl_is_binding(v)) {
                    jl_binding_t *b = (jl_binding_t*)v;
                    write_pointerfield(s, (jl_value_t*)b->globalref->mod);
                    write_pointerfield(s, (jl_value_t*)b->globalref->name);
                    continue;
                }
                else if (jl_is_method_instance(v)) {
                    assert(f == s->s);
                    jl_method_instance_t *mi = (jl_method_instance_t*)v;
                    write_pointerfield(s, mi->def.value);
                    write_pointerfield(s, mi->specTypes);
                    write_pointerfield(s, (jl_value_t*)mi->sparam_vals);
                    continue;
                }
                else if (jl_is_datatype(v)) {
                    for (size_t i = 0; i < s->uniquing_super.len; i++) {
                        if (s->uniquing_super.items[i] == (void*)v) {
                            s->uniquing_super.items[i] = arraylist_pop(&s->uniquing_super);
                            arraylist_push(&s->uniquing_types, (void*)(uintptr_t)(reloc_offset|3));
                        }
                    }
                }
                else {
                    assert(jl_is_datatype_singleton(t) && "unreachable");
                }
            }
            else if (needs_recaching(v, s->query_cache)) {
                arraylist_push(jl_is_datatype(v) ? &s->fixup_types : &s->fixup_objs, (void*)reloc_offset);
            }
        }

        // write data
        if (jl_is_array(v)) {
            assert(f == s->s);
            // Internal data for types in julia.h with `jl_array_t` field(s)
            jl_array_t *ar = (jl_array_t*)v;
            // copy header
            size_t headersize = sizeof(jl_array_t) + jl_array_ndims(ar)*sizeof(size_t);
            ios_write(f, (char*)v, headersize);
            // make some header modifications in-place
            jl_array_t *newa = (jl_array_t*)&f->buf[reloc_offset];
            newa->ref.mem = NULL; // relocation offset
            arraylist_push(&s->relocs_list, (void*)(reloc_offset + offsetof(jl_array_t, ref.mem))); // relocation location
            jl_value_t *mem = get_replaceable_field((jl_value_t**)&ar->ref.mem, 1);
            arraylist_push(&s->relocs_list, (void*)backref_id(s, mem, s->link_ids_relocs)); // relocation target
            record_memoryref(s, reloc_offset + offsetof(jl_array_t, ref), ar->ref);
        }
        else if (jl_is_genericmemory(v)) {
            assert(f == s->s);
            // Internal data for types in julia.h with `jl_genericmemory_t` field(s)
            jl_genericmemory_t *m = (jl_genericmemory_t*)v;
            const jl_datatype_layout_t *layout = t->layout;
            size_t len = m->length;
            // if (jl_genericmemory_how(m) == JL_GENERICMEMORY_STRINGOWNED) {
            //     jl_value_t *owner = jl_genericmemory_data_owner_field(m);
            //     write_uint(f, len);
            //     write_pointerfield(s, owner);
            //     write_pointerfield(s, owner);
            //     jl_genericmemory_t *new_mem = (jl_genericmemory_t*)&f->buf[reloc_offset];
            //     assert(new_mem->ptr == NULL);
            //     new_mem->ptr = (void*)((char*)m->ptr - (char*)owner); // relocation offset
            // }
            // else
            {
                size_t datasize = len * layout->size;
                size_t tot = datasize;
                int isbitsunion = layout->flags.arrayelem_isunion;
                if (isbitsunion)
                    tot += len;
                size_t headersize = sizeof(jl_genericmemory_t);
                // copy header
                ios_write(f, (char*)v, headersize);
                // write data
                if (!layout->flags.arrayelem_isboxed && layout->first_ptr < 0) {
                    // set owner to NULL
                    write_pointer(f);
                    // Non-pointer eltypes get encoded in the const_data section
                    size_t alignment_amt = JL_SMALL_BYTE_ALIGNMENT;
                    if (tot >= ARRAY_CACHE_ALIGN_THRESHOLD)
                        alignment_amt = JL_CACHE_BYTE_ALIGNMENT;
                    uintptr_t data = LLT_ALIGN(ios_pos(s->const_data), alignment_amt);
                    write_padding(s->const_data, data - ios_pos(s->const_data));
                    // write data and relocations
                    jl_genericmemory_t *new_mem = (jl_genericmemory_t*)&f->buf[reloc_offset];
                    new_mem->ptr = NULL; // relocation offset
                    data /= sizeof(void*);
                    assert(data < ((uintptr_t)1 << RELOC_TAG_OFFSET) && "offset to constant data too large");
                    arraylist_push(&s->relocs_list, (void*)(reloc_offset + offsetof(jl_genericmemory_t, ptr))); // relocation location
                    arraylist_push(&s->relocs_list, (void*)(((uintptr_t)ConstDataRef << RELOC_TAG_OFFSET) + data)); // relocation target
                    jl_value_t *et = jl_tparam1(t);
                    if (jl_is_cpointer_type(et)) {
                        // reset Ptr fields to C_NULL (but keep MAP_FAILED / INVALID_HANDLE)
                        const intptr_t *data = (const intptr_t*)m->ptr;
                        size_t i;
                        for (i = 0; i < len; i++) {
                            if (data[i] != -1)
                                write_pointer(s->const_data);
                            else
                                ios_write(s->const_data, (char*)&data[i], sizeof(data[i]));
                        }
                    }
                    else {
                        if (isbitsunion) {
                            ios_write(s->const_data, (char*)m->ptr, datasize);
                            ios_write(s->const_data, jl_genericmemory_typetagdata(m), len);
                        }
                        else {
                            ios_write(s->const_data, (char*)m->ptr, tot);
                        }
                    }
                    if (len == 0) { // TODO: should we have a zero-page, instead of writing each type's fragment separately?
                        write_padding(s->const_data, layout->size ? layout->size : isbitsunion);
                    }
                    else if (jl_genericmemory_how(m) == JL_GENERICMEMORY_STRINGOWNED) {
                        assert(jl_is_string(jl_genericmemory_data_owner_field(m)));
                        write_padding(s->const_data, 1);
                    }
                }
                else {
                    // Pointer eltypes are encoded in the mutable data section
                    headersize = LLT_ALIGN(headersize, JL_SMALL_BYTE_ALIGNMENT);
                    size_t data = LLT_ALIGN(ios_pos(f), JL_SMALL_BYTE_ALIGNMENT);
                    write_padding(f, data - ios_pos(f));
                    assert(reloc_offset + headersize == ios_pos(f));
                    jl_genericmemory_t *new_mem = (jl_genericmemory_t*)&f->buf[reloc_offset];
                    new_mem->ptr = (void*)headersize; // relocation offset
                    arraylist_push(&s->relocs_list, (void*)(reloc_offset + offsetof(jl_genericmemory_t, ptr))); // relocation location
                    arraylist_push(&s->relocs_list, (void*)(((uintptr_t)DataRef << RELOC_TAG_OFFSET) + item)); // relocation target
                    if (!layout->flags.arrayelem_isboxed) {
                        // copy all of the data first
                        const char *data = (const char*)m->ptr;
                        ios_write(f, data, datasize);
                        // the rewrite all of the embedded pointers to null+relocation
                        uint16_t elsz = layout->size;
                        size_t j, np = layout->first_ptr < 0 ? 0 : layout->npointers;
                        size_t i;
                        for (i = 0; i < len; i++) {
                            for (j = 0; j < np; j++) {
                                size_t offset = i * elsz + jl_ptr_offset(t, j) * sizeof(jl_value_t*);
                                jl_value_t *fld = get_replaceable_field((jl_value_t**)&data[offset], 1);
                                size_t fld_pos = reloc_offset + headersize + offset;
                                if (fld != NULL) {
                                    arraylist_push(&s->relocs_list, (void*)(uintptr_t)fld_pos); // relocation location
                                    arraylist_push(&s->relocs_list, (void*)backref_id(s, fld, s->link_ids_relocs)); // relocation target
                                    record_uniquing(s, fld, fld_pos);
                                }
                                memset(&f->buf[fld_pos], 0, sizeof(fld)); // relocation offset (none)
                            }
                        }
                    }
                    else {
                        jl_value_t **data = (jl_value_t**)m->ptr;
                        size_t i;
                        for (i = 0; i < len; i++) {
                            jl_value_t *e = get_replaceable_field(&data[i], 1);
                            write_pointerfield(s, e);
                        }
                    }
                }
            }
        }
        else if (jl_typeis(v, jl_module_type)) {
            assert(f == s->s);
            jl_write_module(s, item, (jl_module_t*)v);
        }
        else if (jl_typetagis(v, jl_task_tag << 4)) {
            abort(); // unreachable
        }
        else if (jl_is_svec(v)) {
            assert(f == s->s);
            ios_write(f, (char*)v, sizeof(void*));
            size_t ii, l = jl_svec_len(v);
            assert(l > 0 || (jl_svec_t*)v == jl_emptysvec);
            for (ii = 0; ii < l; ii++) {
                write_pointerfield(s, jl_svecref(v, ii));
            }
        }
        else if (jl_is_string(v)) {
            ios_write(f, (char*)v, sizeof(void*) + jl_string_len(v));
            write_uint8(f, '\0'); // null-terminated strings for easier C-compatibility
        }
        else if (jl_is_foreign_type(t) == 1) {
            abort(); // unreachable
        }
        else if (jl_datatype_nfields(t) == 0) {
            // The object has no fields, so we just snapshot its byte representation
            assert(t->layout->npointers == 0);
            ios_write(f, (char*)v, jl_datatype_size(t));
        }
        else if (jl_bigint_type && jl_typetagis(v, jl_bigint_type)) {
            // foreign types require special handling
            assert(f == s->s);
            jl_value_t *sizefield = jl_get_nth_field(v, 1);
            int32_t sz = jl_unbox_int32(sizefield);
            int32_t nw = (sz == 0 ? 1 : (sz < 0 ? -sz : sz));
            size_t nb = nw * gmp_limb_size;
            ios_write(f, (char*)&nw, sizeof(int32_t));
            ios_write(f, (char*)&sz, sizeof(int32_t));
            uintptr_t data = LLT_ALIGN(ios_pos(s->const_data), 8);
            write_padding(s->const_data, data - ios_pos(s->const_data));
            data /= sizeof(void*);
            assert(data < ((uintptr_t)1 << RELOC_TAG_OFFSET) && "offset to constant data too large");
            arraylist_push(&s->relocs_list, (void*)(reloc_offset + 8)); // relocation location
            arraylist_push(&s->relocs_list, (void*)(((uintptr_t)ConstDataRef << RELOC_TAG_OFFSET) + data)); // relocation target
            void *pdata = jl_unbox_voidpointer(jl_get_nth_field(v, 2));
            ios_write(s->const_data, (char*)pdata, nb);
            write_pointer(f);
        }
        else {
            // Generic object::DataType serialization by field
            const char *data = (const char*)v;
            size_t i, nf = jl_datatype_nfields(t);
            size_t tot = 0;
            for (i = 0; i < nf; i++) {
                size_t offset = jl_field_offset(t, i);
                const char *slot = data + offset;
                write_padding(f, offset - tot);
                tot = offset;
                size_t fsz = jl_field_size(t, i);
                jl_value_t *replace = (jl_value_t*)ptrhash_get(&bits_replace, (void*)slot);
                if (replace != HT_NOTFOUND && fsz > 0) {
                    assert(t->name->mutabl && !jl_field_isptr(t, i));
                    jl_value_t *rty = jl_typeof(replace);
                    size_t sz = jl_datatype_size(rty);
                    ios_write(f, (const char*)replace, sz);
                    jl_value_t *ft = jl_field_type_concrete(t, i);
                    int isunion = jl_is_uniontype(ft);
                    unsigned nth = 0;
                    if (!jl_find_union_component(ft, rty, &nth))
                        assert(0 && "invalid field assignment to isbits union");
                    assert(sz <= fsz - isunion);
                    write_padding(f, fsz - sz - isunion);
                    if (isunion)
                        write_uint8(f, nth);
                }
                else if (t->name->mutabl && jl_is_cpointer_type(jl_field_type_concrete(t, i)) && *(intptr_t*)slot != -1) {
                    // reset Ptr fields to C_NULL (but keep MAP_FAILED / INVALID_HANDLE)
                    assert(!jl_field_isptr(t, i));
                    write_pointer(f);
                }
                else if (fsz > 0) {
                    ios_write(f, slot, fsz);
                }
                tot += fsz;
            }

            size_t np = t->layout->npointers;
            size_t fldidx = 1;
            for (i = 0; i < np; i++) {
                size_t offset = jl_ptr_offset(t, i) * sizeof(jl_value_t*);
                while (offset >= (fldidx == nf ? jl_datatype_size(t) : jl_field_offset(t, fldidx)))
                    fldidx++;
                int mutabl = !jl_field_isconst(t, fldidx - 1);
                jl_value_t *fld = get_replaceable_field((jl_value_t**)&data[offset], mutabl);
                size_t fld_pos = offset + reloc_offset;
                if (fld != NULL) {
                    arraylist_push(&s->relocs_list, (void*)(uintptr_t)(fld_pos)); // relocation location
                    arraylist_push(&s->relocs_list, (void*)backref_id(s, fld, s->link_ids_relocs)); // relocation target
                    record_uniquing(s, fld, fld_pos);
                }
                memset(&f->buf[fld_pos], 0, sizeof(fld)); // relocation offset (none)
            }

            // Need do a tricky fieldtype walk an record all memoryref we find inlined in this value
            record_memoryrefs_inside(s, t, reloc_offset, data);

            // A few objects need additional handling beyond the generic serialization above
            if (s->incremental && jl_typetagis(v, jl_typemap_entry_type)) {
                assert(f == s->s);
                jl_typemap_entry_t *newentry = (jl_typemap_entry_t*)&s->s->buf[reloc_offset];
                if (jl_atomic_load_relaxed(&newentry->max_world) == ~(size_t)0) {
                    if (jl_atomic_load_relaxed(&newentry->min_world) > 1) {
                        jl_atomic_store_relaxed(&newentry->min_world, ~(size_t)0);
                        jl_atomic_store_relaxed(&newentry->max_world, WORLD_AGE_REVALIDATION_SENTINEL);
                        arraylist_push(&s->fixup_objs, (void*)reloc_offset);
                    }
                }
                else {
                    // garbage newentry - delete it :(
                    jl_atomic_store_relaxed(&newentry->min_world, 1);
                    jl_atomic_store_relaxed(&newentry->max_world, 0);
                }
            }
            else if (s->incremental && jl_is_binding_partition(v)) {
                jl_binding_partition_t *newbpart = (jl_binding_partition_t*)&s->s->buf[reloc_offset];
                size_t max_world = jl_atomic_load_relaxed(&newbpart->max_world);
                if (max_world == ~(size_t)0) {
                    // Still valid. Will be considered to be defined in jl_require_world
                    // after reload, which is the first world before new code runs.
                    // We use this as a quick check to determine whether a binding was
                    // invalidated. If a binding was first defined in or before
                    // jl_require_world, then we can assume that all precompile processes
                    // will have seen it consistently.
                    jl_atomic_store_relaxed(&newbpart->min_world, jl_require_world);
                }
                else {
                    // The world will not be reachable after loading
                    jl_atomic_store_relaxed(&newbpart->min_world, 1);
                    jl_atomic_store_relaxed(&newbpart->max_world, 0);
                }
            }
            else if (jl_is_method(v)) {
                assert(f == s->s);
                write_padding(f, sizeof(jl_method_t) - tot); // hidden fields
                jl_method_t *m = (jl_method_t*)v;
                jl_method_t *newm = (jl_method_t*)&f->buf[reloc_offset];
                if (s->incremental) {
                    if (jl_atomic_load_relaxed(&newm->primary_world) > 1) {
                        jl_atomic_store_relaxed(&newm->primary_world, ~(size_t)0); // min-world
                        int dispatch_status = jl_atomic_load_relaxed(&newm->dispatch_status);
                        int new_dispatch_status = 0;
                        if (!(dispatch_status & METHOD_SIG_LATEST_ONLY))
                            new_dispatch_status |= METHOD_SIG_PRECOMPILE_MANY;
                        jl_atomic_store_relaxed(&newm->dispatch_status, new_dispatch_status);
                        arraylist_push(&s->fixup_objs, (void*)reloc_offset);
                    }
                }
                else {
                    newm->nroots_sysimg = m->roots ? jl_array_len(m->roots) : 0;
                }
            }
            else if (jl_is_method_instance(v)) {
                assert(f == s->s);
                jl_method_instance_t *newmi = (jl_method_instance_t*)&f->buf[reloc_offset];
                jl_atomic_store_relaxed(&newmi->flags, 0);
                if (s->incremental) {
                    jl_atomic_store_relaxed(&newmi->dispatch_status, 0);
                }
            }
            else if (jl_is_code_instance(v)) {
                assert(f == s->s);

                // Handle the native-code pointers
                jl_code_instance_t *ci = (jl_code_instance_t*)v;
                jl_code_instance_t *newci = (jl_code_instance_t*)&f->buf[reloc_offset];

                if (s->incremental) {
                    if (jl_atomic_load_relaxed(&ci->max_world) == ~(size_t)0) {
                        //assert(jl_atomic_load_relaxed(&ci->edges) != jl_emptysvec); // some code (such as !==) might add a method lookup restriction but not keep the edges
                        jl_atomic_store_release(&newci->min_world, ~(size_t)0);
                        jl_atomic_store_release(&newci->max_world, WORLD_AGE_REVALIDATION_SENTINEL);
                        arraylist_push(&s->fixup_objs, (void*)reloc_offset);
                    }
                    else {
                        // garbage object - delete it :(
                        jl_atomic_store_release(&newci->min_world, 1);
                        jl_atomic_store_release(&newci->max_world, 0);
                    }
                }
                jl_atomic_store_relaxed(&newci->time_compile, 0.0);
                jl_atomic_store_relaxed(&newci->invoke, NULL);
                // preserve only JL_CI_FLAGS_NATIVE_CACHE_VALID bits
                jl_atomic_store_relaxed(&newci->flags, jl_atomic_load_relaxed(&newci->flags) & JL_CI_FLAGS_NATIVE_CACHE_VALID);
                jl_atomic_store_relaxed(&newci->specptr.fptr, NULL);
                int8_t fptr_id = JL_API_NULL;
                int8_t builtin_id = 0;
                if (jl_atomic_load_relaxed(&ci->invoke) == jl_fptr_const_return) {
                    fptr_id = JL_API_CONST;
                }
                else {
                    if (jl_is_method(jl_get_ci_mi(ci)->def.method)) {
                        builtin_id = jl_fptr_id(jl_atomic_load_relaxed(&ci->specptr.fptr));
                        if (builtin_id) { // found in the table of builtins
                            assert(builtin_id >= 2);
                            fptr_id = JL_API_BUILTIN;
                        }
                        else {
                            int32_t invokeptr_id = 0;
                            int32_t specfptr_id = 0;
                            jl_get_function_id(native_functions, ci, &invokeptr_id, &specfptr_id); // see if we generated code for it
                            if (invokeptr_id) {
                                if (invokeptr_id == -1) {
                                    fptr_id = JL_API_BOXED;
                                }
                                else if (invokeptr_id == -2) {
                                    fptr_id = JL_API_WITH_PARAMETERS;
                                }
                                else if (invokeptr_id == -3) {
                                    abort();
                                }
                                else if (invokeptr_id == -4) {
                                    fptr_id = JL_API_OC_CALL;
                                }
                                else if (invokeptr_id == -5) {
                                    abort();
                                }
                                else {
                                    assert(invokeptr_id > 0);
                                    ios_ensureroom(s->fptr_record, invokeptr_id * sizeof(void*));
                                    ios_seek(s->fptr_record, (invokeptr_id - 1) * sizeof(void*));
                                    write_reloc_t(s->fptr_record, (reloc_t)~reloc_offset);
#ifdef _P64
                                    if (sizeof(reloc_t) < 8)
                                        write_padding(s->fptr_record, 8 - sizeof(reloc_t));
#endif
                                }
                                if (specfptr_id) {
                                    assert(specfptr_id > invokeptr_id && specfptr_id > 0);
                                    ios_ensureroom(s->fptr_record, specfptr_id * sizeof(void*));
                                    ios_seek(s->fptr_record, (specfptr_id - 1) * sizeof(void*));
                                    write_reloc_t(s->fptr_record, reloc_offset);
#ifdef _P64
                                    if (sizeof(reloc_t) < 8)
                                        write_padding(s->fptr_record, 8 - sizeof(reloc_t));
#endif
                                }
                            }
                        }
                    }
                }
                jl_atomic_store_relaxed(&newci->invoke, NULL); // relocation offset
                if (fptr_id != JL_API_NULL) {
                    assert(fptr_id < BuiltinFunctionTag && "too many functions to serialize");
                    arraylist_push(&s->relocs_list, (void*)(reloc_offset + offsetof(jl_code_instance_t, invoke))); // relocation location
                    arraylist_push(&s->relocs_list, (void*)(((uintptr_t)FunctionRef << RELOC_TAG_OFFSET) + fptr_id)); // relocation target
                }
                if (builtin_id >= 2) {
                    arraylist_push(&s->relocs_list, (void*)(reloc_offset + offsetof(jl_code_instance_t, specptr.fptr))); // relocation location
                    arraylist_push(&s->relocs_list, (void*)(((uintptr_t)FunctionRef << RELOC_TAG_OFFSET) + BuiltinFunctionTag + builtin_id - 2)); // relocation target
                }
            }
            else if (jl_is_datatype(v)) {
                assert(f == s->s);
                jl_datatype_t *dt = (jl_datatype_t*)v;
                jl_datatype_t *newdt = (jl_datatype_t*)&f->buf[reloc_offset];

                if (dt->layout != NULL) {
                    size_t nf = dt->layout->nfields;
                    size_t np = dt->layout->npointers;
                    size_t fieldsize = 0;
                    uint8_t is_foreign_type = dt->layout->flags.fielddesc_type == 3;
                    if (!is_foreign_type) {
                        fieldsize = jl_fielddesc_size(dt->layout->flags.fielddesc_type);
                    }
                    char *flddesc = (char*)dt->layout;
                    size_t fldsize = sizeof(jl_datatype_layout_t) + nf * fieldsize;
                    if (!is_foreign_type && dt->layout->first_ptr != -1)
                        fldsize += np << dt->layout->flags.fielddesc_type;
                    uintptr_t layout = LLT_ALIGN(ios_pos(s->const_data), sizeof(void*));
                    write_padding(s->const_data, layout - ios_pos(s->const_data)); // realign stream
                    newdt->layout = NULL; // relocation offset
                    layout /= sizeof(void*);
                    arraylist_push(&s->relocs_list, (void*)(reloc_offset + offsetof(jl_datatype_t, layout))); // relocation location
                    arraylist_push(&s->relocs_list, (void*)(((uintptr_t)ConstDataRef << RELOC_TAG_OFFSET) + layout)); // relocation target
                    ios_write(s->const_data, flddesc, fldsize);
                    if (is_foreign_type) {
                        // make sure we have space for the extra hidden pointers
                        // zero them since they will need to be re-initialized externally
                        assert(fldsize == sizeof(jl_datatype_layout_t));
                        jl_fielddescdyn_t dyn = {0, 0};
                        ios_write(s->const_data, (char*)&dyn, sizeof(jl_fielddescdyn_t));
                    }
                }
                void *superidx = ptrhash_get(&serialization_order, dt->super);
                if (s->incremental && superidx != HT_NOTFOUND && from_seroder_entry(superidx) > item && needs_uniquing((jl_value_t*)dt->super, s->query_cache))
                    arraylist_push(&s->uniquing_super, dt->super);
            }
            else if (jl_is_typename(v)) {
                assert(f == s->s);
                jl_typename_t *tn = (jl_typename_t*)v;
                jl_typename_t *newtn = (jl_typename_t*)&f->buf[reloc_offset];
                if (tn->atomicfields != NULL) {
                    size_t nb = (jl_svec_len(tn->names) + 31) / 32 * sizeof(uint32_t);
                    uintptr_t layout = LLT_ALIGN(ios_pos(s->const_data), sizeof(void*));
                    write_padding(s->const_data, layout - ios_pos(s->const_data)); // realign stream
                    newtn->atomicfields = NULL; // relocation offset
                    layout /= sizeof(void*);
                    arraylist_push(&s->relocs_list, (void*)(reloc_offset + offsetof(jl_typename_t, atomicfields))); // relocation location
                    arraylist_push(&s->relocs_list, (void*)(((uintptr_t)ConstDataRef << RELOC_TAG_OFFSET) + layout)); // relocation target
                    ios_write(s->const_data, (char*)tn->atomicfields, nb);
                }
                if (tn->constfields != NULL) {
                    size_t nb = (jl_svec_len(tn->names) + 31) / 32 * sizeof(uint32_t);
                    uintptr_t layout = LLT_ALIGN(ios_pos(s->const_data), sizeof(void*));
                    write_padding(s->const_data, layout - ios_pos(s->const_data)); // realign stream
                    newtn->constfields = NULL; // relocation offset
                    layout /= sizeof(void*);
                    arraylist_push(&s->relocs_list, (void*)(reloc_offset + offsetof(jl_typename_t, constfields))); // relocation location
                    arraylist_push(&s->relocs_list, (void*)(((uintptr_t)ConstDataRef << RELOC_TAG_OFFSET) + layout)); // relocation target
                    ios_write(s->const_data, (char*)tn->constfields, nb);
                }
            }
            else if (jl_is_globalref(v)) {
                assert(f == s->s);
                jl_globalref_t *gr = (jl_globalref_t*)v;
                if (s->incremental && jl_object_in_image((jl_value_t*)gr->mod)) {
                    // will need to populate the binding field later
                    arraylist_push(&s->fixup_objs, (void*)reloc_offset);
                }
            }
            else if (jl_is_genericmemoryref(v)) {
                assert(f == s->s);
                record_memoryref(s, reloc_offset, *(jl_genericmemoryref_t*)v);
            }
            else {
                write_padding(f, jl_datatype_size(t) - tot);
            }
        }
    }
    assert(s->uniquing_super.len == 0);
}

// In deserialization, create Symbols and set up the
// index for backreferencing
static void jl_read_symbols(jl_serializer_state *s)
{
    assert(deser_sym.len == 0);
    uintptr_t base = (uintptr_t)&s->symbols->buf[0];
    uintptr_t end = base + s->symbols->size;
    while (base < end) {
        uint32_t len = jl_load_unaligned_i32((void*)base);
        base += 4;
        const char *str = (const char*)base;
        base += len + 1;
        //printf("symbol %3d: %s\n", len, str);
        jl_sym_t *sym = _jl_symbol(str, len);
        arraylist_push(&deser_sym, (void*)sym);
    }
}


// In serialization, extract the appropriate serializer position for RefTags-encoded index `reloc_item`.
// Used for hard-coded tagged items, `relocs_list`, and `gctags_list`
static uintptr_t get_reloc_for_item(uintptr_t reloc_item, size_t reloc_offset)
{
    enum RefTags tag = (enum RefTags)(reloc_item >> RELOC_TAG_OFFSET);
    if (tag == DataRef) {
        // first serialized segment
        // need to compute the final relocation offset via the layout table
        assert(reloc_item < layout_table.len);
        uintptr_t reloc_base = (uintptr_t)layout_table.items[reloc_item];
        assert(reloc_base != 0 && "layout offset missing for relocation item");
        if (reloc_base & 1) {
            // convert to a ConstDataRef
            tag = ConstDataRef;
            reloc_base &= ~(uintptr_t)1;
            assert(LLT_ALIGN(reloc_base, sizeof(void*)) == reloc_base);
            reloc_base /= sizeof(void*);
            assert(reloc_offset == 0);
        }
        // write reloc_offset into s->s at pos
        return ((uintptr_t)tag << RELOC_TAG_OFFSET) + reloc_base + reloc_offset;
    }
    else {
        // just write the item reloc_id directly
#ifndef JL_NDEBUG
        assert(reloc_offset == 0 && "offsets for relocations to builtin objects should be precomposed in the reloc_item");
        size_t offset = (reloc_item & (((uintptr_t)1 << RELOC_TAG_OFFSET) - 1));
        switch (tag) {
        case ConstDataRef:
            break;
        case SymbolRef:
            assert(offset < nsym_tag && "corrupt relocation item id");
            break;
        case TagRef:
            assert(offset < 2 * NBOX_C + 258 && "corrupt relocation item id");
            break;
        case FunctionRef:
            if (offset & BuiltinFunctionTag) {
                offset &= ~BuiltinFunctionTag;
                assert(offset < jl_n_builtins && "unknown function pointer id");
            }
            else {
                assert(offset < JL_API_MAX && "unknown function pointer id");
            }
            break;
        case SysimageLinkage:
            break;
        case ExternalLinkage:
            break;
        default:
            assert(0 && "corrupt relocation item id");
            abort();
        }
#endif
        return reloc_item; // pre-composed relocation + offset
    }
}

// Compute target location at deserialization
static inline uintptr_t get_item_for_reloc(jl_serializer_state *s, uintptr_t base, uintptr_t reloc_id, jl_array_t *link_ids, int *link_index) JL_NOTSAFEPOINT
{
    enum RefTags tag = (enum RefTags)(reloc_id >> RELOC_TAG_OFFSET);
    size_t offset = (reloc_id & (((uintptr_t)1 << RELOC_TAG_OFFSET) - 1));
    switch (tag) {
    case DataRef:
        assert(offset <= s->s->size);
        return (uintptr_t)base + offset;
    case ConstDataRef:
        offset *= sizeof(void*);
        assert(offset <= s->const_data->size);
        return (uintptr_t)s->const_data->buf + offset;
    case SymbolRef:
        assert(offset < deser_sym.len && deser_sym.items[offset] && "corrupt relocation item id");
        return (uintptr_t)deser_sym.items[offset];
    case TagRef:
        if (offset == 0)
            return (uintptr_t)s->ptls->root_task;
        if (offset == 1)
            return (uintptr_t)jl_nothing;
        offset -= 2;
        if (offset < NBOX_C)
            return (uintptr_t)jl_box_int64((int64_t)offset - NBOX_C / 2);
        offset -= NBOX_C;
        if (offset < NBOX_C)
            return (uintptr_t)jl_box_int32((int32_t)offset - NBOX_C / 2);
        offset -= NBOX_C;
        if (offset < 256)
            return (uintptr_t)jl_box_uint8(offset);
        // offset -= 256;
        assert(0 && "corrupt relocation item id");
        jl_unreachable(); // terminate control flow if assertion is disabled.
    case FunctionRef:
        if (offset & BuiltinFunctionTag) {
            offset &= ~BuiltinFunctionTag;
            assert(offset < jl_n_builtins && "unknown function pointer ID");
            return (uintptr_t)jl_builtin_f_addrs[offset];
        }
        switch ((jl_callingconv_t)offset) {
        case JL_API_BOXED:
            if (s->image->fptrs.nptrs)
                return (uintptr_t)jl_fptr_args;
            return (uintptr_t)NULL;
        case JL_API_WITH_PARAMETERS:
            if (s->image->fptrs.nptrs)
                return (uintptr_t)jl_fptr_sparam;
            return (uintptr_t)NULL;
        case JL_API_OC_CALL:
            if (s->image->fptrs.nptrs)
                return (uintptr_t)jl_f_opaque_closure_call;
            return (uintptr_t)NULL;
        case JL_API_CONST:
            return (uintptr_t)jl_fptr_const_return;
        case JL_API_INTERPRETED:
            return (uintptr_t)jl_fptr_interpret_call;
        case JL_API_BUILTIN:
            return (uintptr_t)jl_fptr_args;
        case JL_API_NULL:
        case JL_API_MAX:
        //default:
            assert("corrupt relocation item id");
        }
    case SysimageLinkage: {
#ifdef _P64
        size_t depsidx = offset >> DEPS_IDX_OFFSET;
        offset &= ((size_t)1 << DEPS_IDX_OFFSET) - 1;
#else
        size_t depsidx = 0;
#endif
        if (relink_active(s, depsidx))
            return relink_resolve(s, depsidx, offset);
        assert(s->buildid_depmods_idxs && depsidx < jl_array_len(s->buildid_depmods_idxs));
        size_t i = jl_array_data(s->buildid_depmods_idxs, uint32_t)[depsidx];
        assert(2*i < jl_linkage_blobs.len);
        return (uintptr_t)jl_linkage_blobs.items[2*i] + offset*SYS_EXTERNAL_LINK_UNIT;
    }
    case ExternalLinkage: {
        assert(link_ids);
        assert(link_index);
        assert(0 <= *link_index && *link_index < jl_array_len(link_ids));
        uint32_t depsidx = jl_array_data(link_ids, uint32_t)[*link_index];
        *link_index += 1;
        if (relink_active(s, depsidx))
            return relink_resolve(s, depsidx, offset);
        assert(depsidx < jl_array_len(s->buildid_depmods_idxs));
        size_t i = jl_array_data(s->buildid_depmods_idxs, uint32_t)[depsidx];
        assert(2*i < jl_linkage_blobs.len);
        return (uintptr_t)jl_linkage_blobs.items[2*i] + offset*SYS_EXTERNAL_LINK_UNIT;
    }
    }
    abort();
}


static void jl_finish_relocs(char *base, size_t size, arraylist_t *list)
{
    for (size_t i = 0; i < list->len; i += 2) {
        size_t pos = (size_t)list->items[i];
        size_t item = (size_t)list->items[i + 1];   // item is tagref-encoded
        uintptr_t *pv = (uintptr_t*)(base + pos);
        assert(pos < size && pos != 0);
        *pv = get_reloc_for_item(item, *pv);
    }
}

static void jl_write_offsetlist(ios_t *s, size_t size, arraylist_t *list)
{
    for (size_t i = 0; i < list->len; i += 2) {
        size_t last_pos = i ? (size_t)list->items[i - 2] : 0;
        size_t pos = (size_t)list->items[i];
        assert(pos < size && pos != 0);
        // write pos as compressed difference.
        size_t pos_diff = pos - last_pos;
        while (pos_diff) {
            assert(pos_diff >= 0);
            if (pos_diff <= 127) {
                write_int8(s, pos_diff);
                break;
            }
            else {
                // Extract the next 7 bits
                int8_t ns = pos_diff & (int8_t)0x7F;
                pos_diff >>= 7;
                // Set the high bit if there's still more
                ns |= (!!pos_diff) << 7;
                write_int8(s, ns);
            }
        }
    }
    write_int8(s, 0);
}


static void jl_write_arraylist(ios_t *s, arraylist_t *list)
{
    write_uint(s, list->len);
    ios_write(s, (const char*)list->items, list->len * sizeof(void*));
}

static void jl_read_reloclist(jl_serializer_state *s, jl_array_t *link_ids, uint8_t bits)
{
    uintptr_t base = (uintptr_t)s->s->buf;
    uintptr_t last_pos = 0;
    uint8_t *current = (uint8_t *)(s->relocs->buf + s->relocs->bpos);
    int link_index = 0;
    while (1) {
        // Read the offset of the next object
        size_t pos_diff = 0;
        size_t cnt = 0;
        while (1) {
            assert(s->relocs->bpos <= s->relocs->size);
            assert((char *)current <= (char *)(s->relocs->buf + s->relocs->size));
            int8_t c = *current++;
            s->relocs->bpos += 1;

            pos_diff |= ((size_t)c & 0x7F) << (7 * cnt++);
            if ((c >> 7) == 0)
                break;
        }
        if (pos_diff == 0)
            break;

        uintptr_t pos = last_pos + pos_diff;
        last_pos = pos;
        uintptr_t *pv = (uintptr_t *)(base + pos);
        uintptr_t v = *pv;
        v = get_item_for_reloc(s, base, v, link_ids, &link_index);
        if (bits && v && ((jl_datatype_t*)v)->smalltag)
            v = (uintptr_t)((jl_datatype_t*)v)->smalltag << 4; // TODO: should we have a representation that supports sweep without a relocation step?
        *pv = v | bits;
    }
    assert(!link_ids || link_index == jl_array_len(link_ids));
}

static void jl_read_memreflist(jl_serializer_state *s)
{
    uintptr_t base = (uintptr_t)s->s->buf;
    uintptr_t last_pos = 0;
    uint8_t *current = (uint8_t *)(s->relocs->buf + s->relocs->bpos);
    while (1) {
        // Read the offset of the next object
        size_t pos_diff = 0;
        size_t cnt = 0;
        while (1) {
            assert(s->relocs->bpos <= s->relocs->size);
            assert((char *)current <= (char *)(s->relocs->buf + s->relocs->size));
            int8_t c = *current++;
            s->relocs->bpos += 1;

            pos_diff |= ((size_t)c & 0x7F) << (7 * cnt++);
            if ((c >> 7) == 0)
                break;
        }
        if (pos_diff == 0)
            break;

        uintptr_t pos = last_pos + pos_diff;
        last_pos = pos;
        jl_genericmemoryref_t *pv = (jl_genericmemoryref_t*)(base + pos);
        size_t offset = (size_t)pv->ptr_or_offset;
        pv->ptr_or_offset = (void*)((char*)pv->mem->ptr + offset);
    }
}


static void jl_read_arraylist(ios_t *s, arraylist_t *list)
{
    size_t list_len = read_uint(s);
    arraylist_new(list, 0);
    arraylist_grow(list, list_len);
    ios_read(s, (char*)list->items, list_len * sizeof(void*));
}

void gc_sweep_sysimg(void) JL_NOTSAFEPOINT
{
    size_t nblobs = n_linkage_blobs();
    if (nblobs == 0)
        return;
    assert(jl_linkage_blobs.len == 2*nblobs);
    assert(jl_image_relocs.len == nblobs);
    for (size_t i = 0; i < 2*nblobs; i+=2) {
        reloc_t *relocs = (reloc_t*)jl_image_relocs.items[i>>1];
        if (!relocs)
            continue;
        uintptr_t base = (uintptr_t)jl_linkage_blobs.items[i];
        uintptr_t last_pos = 0;
        uint8_t *current = (uint8_t *)relocs;
        while (1) {
            // Read the offset of the next object
            size_t pos_diff = 0;
            size_t cnt = 0;
            while (1) {
                int8_t c = *current++;
                pos_diff |= ((size_t)c & 0x7F) << (7 * cnt++);
                if ((c >> 7) == 0)
                    break;
            }
            if (pos_diff == 0)
                break;

            uintptr_t pos = last_pos + pos_diff;
            last_pos = pos;
            jl_taggedvalue_t *o = (jl_taggedvalue_t *)(base + pos);
            o->bits.gc = GC_OLD;
            assert(o->bits.in_image == 1);
        }
    }
}

// jl_write_value and jl_read_value are used for storing Julia objects that are adjuncts to
// the image proper. For example, new methods added to external callables require
// insertion into the appropriate method table.
#define jl_write_value(s, v) _jl_write_value((s), (jl_value_t*)(v))
static void _jl_write_value(jl_serializer_state *s, jl_value_t *v) JL_GC_DISABLED
{
    if (v == NULL) {
        write_reloc_t(s->s, 0);
        return;
    }
    uintptr_t item = backref_id(s, v, NULL);
    uintptr_t reloc = get_reloc_for_item(item, 0);
    write_reloc_t(s->s, reloc);
}

static jl_value_t *jl_read_value(jl_serializer_state *s)
{
    uintptr_t base = (uintptr_t)s->s->buf;
    uintptr_t offset = *(reloc_t*)(base + (uintptr_t)s->s->bpos);
    s->s->bpos += sizeof(reloc_t);
    if (offset == 0)
        return NULL;
    return (jl_value_t*)get_item_for_reloc(s, base, offset, NULL, NULL);
}

// The next two, `jl_read_offset` and `jl_delayed_reloc`, are essentially a split version
// of `jl_read_value` that allows usage of the relocation data rather than passing NULL
// to `get_item_for_reloc`.
// This works around what would otherwise be an order-dependency conundrum: objects
// that may require relocation data have to be inserted into `serialization_order`,
// and that may include some of the adjunct data that gets serialized via
// `jl_write_value`. But we can't interpret them properly until we read the relocation
// data, and that happens after we pull items out of the serialization stream.
static uintptr_t jl_read_offset(jl_serializer_state *s)
{
    uintptr_t base = (uintptr_t)&s->s->buf[0];
    uintptr_t offset = *(reloc_t*)(base + (uintptr_t)s->s->bpos);
    s->s->bpos += sizeof(reloc_t);
    return offset;
}

static jl_value_t *jl_delayed_reloc(jl_serializer_state *s, uintptr_t offset) JL_GC_DISABLED
{
    if (!offset)
        return NULL;
    uintptr_t base = (uintptr_t)s->s->buf;
    int link_index = 0;
    jl_value_t *ret = (jl_value_t*)get_item_for_reloc(s, base, offset, s->link_ids_relocs, &link_index);
    assert(!s->link_ids_relocs || link_index < jl_array_len(s->link_ids_relocs));
    return ret;
}


static void jl_update_all_fptrs(jl_serializer_state *s, jl_image_t *image)
{
    jl_image_fptrs_t fvars = image->fptrs;
    // make these NULL now so we skip trying to restore GlobalVariable pointers later
    image->gvars_base = NULL;
    if (fvars.nptrs == 0)
        return;

    memcpy(image->jl_small_typeof, &jl_small_typeof, sizeof(jl_small_typeof));

    int img_fvars_max = s->fptr_record->size / sizeof(void*);
    size_t i;
    uintptr_t base = (uintptr_t)&s->s->buf[0];
    // These will become MethodInstance references, but they start out as a list of
    // offsets into `s` for CodeInstances
    jl_method_instance_t **linfos = (jl_method_instance_t**)&s->fptr_record->buf[0];
    uint32_t clone_idx = 0;
    for (i = 0; i < img_fvars_max; i++) {
        reloc_t offset = *(reloc_t*)&linfos[i];
        linfos[i] = NULL;
        if (offset != 0) {
            int specfunc = 1;
            if (offset & ((uintptr_t)1 << (8 * sizeof(reloc_t) - 1))) {
                // if high bit is set, this is the func wrapper, not the specfunc
                specfunc = 0;
                offset = ~offset;
            }
            jl_code_instance_t *codeinst = (jl_code_instance_t*)(base + offset);
            assert(jl_is_method(jl_get_ci_mi(codeinst)->def.method) && jl_atomic_load_relaxed(&codeinst->invoke) != jl_fptr_const_return);
            assert(specfunc ? jl_atomic_load_relaxed(&codeinst->invoke) != NULL : jl_atomic_load_relaxed(&codeinst->invoke) == NULL);
            linfos[i] = jl_get_ci_mi(codeinst);     // now it's a MethodInstance
            void *fptr = fvars.ptrs[i];
            for (; clone_idx < fvars.nclones; clone_idx++) {
                uint32_t idx = fvars.clone_idxs[clone_idx] & jl_sysimg_val_mask;
                if (idx < i)
                    continue;
                if (idx == i)
                    fptr = fvars.clone_ptrs[clone_idx];
                break;
            }
            if (specfunc) {
                jl_atomic_store_relaxed(&codeinst->specptr.fptr, fptr);
                // TODO: set JL_CI_FLAGS_SPECPTR_SPECIALIZED only if confirmed to be true
                jl_atomic_store_relaxed(&codeinst->flags, jl_atomic_load_relaxed(&codeinst->flags) | JL_CI_FLAGS_SPECPTR_SPECIALIZED | JL_CI_FLAGS_INVOKE_MATCHES_SPECPTR | JL_CI_FLAGS_FROM_IMAGE);
            }
            else {
                jl_atomic_store_relaxed(&codeinst->invoke, (jl_callptr_t)fptr);
            }
        }
    }
    // Tell LLVM about the native code
    jl_register_fptrs(image->base, &fvars, linfos, img_fvars_max);
}

static uint32_t write_gvars(jl_serializer_state *s, arraylist_t *globals, arraylist_t *external_fns) JL_GC_DISABLED
{
    size_t len = globals->len + external_fns->len;
    ios_ensureroom(s->gvar_record, len * sizeof(reloc_t));
    for (size_t i = 0; i < globals->len; i++) {
        void *g = globals->items[i];
        uintptr_t item = backref_id(s, g, s->link_ids_gvars);
        uintptr_t reloc = get_reloc_for_item(item, 0);
        write_reloc_t(s->gvar_record, reloc);
        record_uniquing(s, (jl_value_t*)g, ((i << 2) | 2)); // mark as gvar && !tag
    }
    for (size_t i = 0; i < external_fns->len; i++) {
        jl_code_instance_t *ci = (jl_code_instance_t*)external_fns->items[i];
        assert(ci && (jl_atomic_load_relaxed(&ci->flags) & JL_CI_FLAGS_SPECPTR_SPECIALIZED));
        uintptr_t item = backref_id(s, (void*)ci, s->link_ids_external_fnvars);
        uintptr_t reloc = get_reloc_for_item(item, 0);
        write_reloc_t(s->gvar_record, reloc);
    }
    return globals->len;
}

// Pointer relocation for native-code referenced global variables
static void jl_update_all_gvars(jl_serializer_state *s, jl_image_t *image, uint32_t external_fns_begin)
{
    if (image->gvars_base == NULL)
        return;
    uintptr_t base = (uintptr_t)s->s->buf;
    size_t i = 0;
    size_t l = s->gvar_record->size / sizeof(reloc_t);
    reloc_t *gvars = (reloc_t*)&s->gvar_record->buf[0];
    int gvar_link_index = 0;
    int external_fns_link_index = 0;
    assert(l == image->ngvars);
    for (i = 0; i < l; i++) {
        uintptr_t offset = gvars[i];
        uintptr_t v = 0;
        if (i < external_fns_begin) {
            v = get_item_for_reloc(s, base, offset, s->link_ids_gvars, &gvar_link_index);
        }
        else {
            v = get_item_for_reloc(s, base, offset, s->link_ids_external_fnvars, &external_fns_link_index);
        }
        uintptr_t *gv = sysimg_gvars(image->gvars_base, image->gvars_offsets, i);
        *gv = v;
    }
    assert(!s->link_ids_gvars || gvar_link_index == jl_array_len(s->link_ids_gvars));
    assert(!s->link_ids_external_fnvars || external_fns_link_index == jl_array_len(s->link_ids_external_fnvars));
}

static void jl_root_new_gvars(jl_serializer_state *s, jl_image_t *image, uint32_t external_fns_begin)
{
    if (image->gvars_base == NULL)
        return;
    size_t i = 0;
    size_t l = s->gvar_record->size / sizeof(reloc_t);
    for (i = 0; i < l; i++) {
        uintptr_t *gv = sysimg_gvars(image->gvars_base, image->gvars_offsets, i);
        uintptr_t v = *gv;
        if (i < external_fns_begin) {
            if (!jl_is_binding(v))
                v = (uintptr_t)jl_as_global_root((jl_value_t*)v, 1);
        }
        else {
            jl_code_instance_t *codeinst = (jl_code_instance_t*) v;
            assert(codeinst && (jl_atomic_load_relaxed(&codeinst->flags) & JL_CI_FLAGS_SPECPTR_SPECIALIZED) && jl_atomic_load_relaxed(&codeinst->specptr.fptr));
            v = (uintptr_t)jl_atomic_load_relaxed(&codeinst->specptr.fptr);
        }
        *gv = v;
    }
}

// Code below helps slim down the images by
// removing cached types not referenced in the stream
static jl_svec_t *jl_prune_type_cache_hash(jl_svec_t *cache) JL_GC_DISABLED
{
    size_t l = jl_svec_len(cache), i;
    size_t sz = 0;
    if (l == 0)
        return cache;
    for (i = 0; i < l; i++) {
        jl_value_t *ti = jl_svecref(cache, i);
        if (ti == jl_nothing)
            continue;
        if (ptrhash_get(&serialization_order, ti) == HT_NOTFOUND)
            jl_svecset(cache, i, jl_nothing);
        else
            sz += 1;
    }
    if (sz < HT_N_INLINE)
        sz = HT_N_INLINE;

    void *idx = ptrhash_get(&serialization_order, cache);
    assert(idx != HT_NOTFOUND && idx != (void*)(uintptr_t)-1);
    assert(serialization_queue.items[from_seroder_entry(idx)] == cache);
    cache = cache_rehash_set(cache, sz);
    // redirect all references to the old cache to relocate to the new cache object
    ptrhash_put(&serialization_order, cache, idx);
    serialization_queue.items[from_seroder_entry(idx)] = cache;
    return cache;
}

static void jl_prune_type_cache_linear(jl_svec_t *cache)
{
    size_t l = jl_svec_len(cache), ins = 0, i;
    for (i = 0; i < l; i++) {
        jl_value_t *ti = jl_svecref(cache, i);
        if (ti == jl_nothing)
            break;
        if (ptrhash_get(&serialization_order, ti) != HT_NOTFOUND)
            jl_svecset(cache, ins++, ti);
    }
    while (ins < l)
        jl_svecset(cache, ins++, jl_nothing);
}

static void jl_prune_mi_backedges(jl_array_t *backedges)
{
    if (backedges == NULL)
        return;
    size_t i = 0, ins = 0, n = jl_array_nrows(backedges);
    while (i < n) {
        jl_value_t *invokeTypes;
        jl_code_instance_t *caller;
        i = get_next_edge(backedges, i, &invokeTypes, &caller);
        if (ptrhash_get(&serialization_order, caller) != HT_NOTFOUND)
            ins = set_next_edge(backedges, ins, invokeTypes, caller);
    }
    jl_array_del_end(backedges, n - ins);
}

static void jl_prune_tn_backedges(jl_array_t *backedges)
{
    size_t i = 0, ins = 0, n = jl_array_nrows(backedges);
    for (i = 1; i < n; i += 2) {
        jl_value_t *ci = jl_array_ptr_ref(backedges, i);
        if (ptrhash_get(&serialization_order, ci) != HT_NOTFOUND) {
            jl_array_ptr_set(backedges, ins++, jl_array_ptr_ref(backedges, i - 1));
            jl_array_ptr_set(backedges, ins++, ci);
        }
    }
    jl_array_del_end(backedges, n - ins);
}

static void jl_prune_mt_backedges(jl_genericmemory_t *allbackedges)
{
    for (size_t i = 0, n = allbackedges->length; i < n; i += 2) {
        jl_value_t *tn = jl_genericmemory_ptr_ref(allbackedges, i);
        jl_value_t *backedges = jl_genericmemory_ptr_ref(allbackedges, i + 1);
        if (tn && tn != jl_nothing && backedges)
            jl_prune_tn_backedges((jl_array_t*)backedges);
    }
}

static void jl_prune_binding_backedges(jl_array_t *backedges)
{
    if (backedges == NULL)
        return;
    size_t i = 0, ins = 0, n = jl_array_nrows(backedges);
    for (i = 0; i < n; i++) {
        jl_value_t *b = jl_array_ptr_ref(backedges, i);
        if (ptrhash_get(&serialization_order, b) != HT_NOTFOUND) {
            jl_array_ptr_set(backedges, ins, b);
            ins++;
        }
    }
    jl_array_del_end(backedges, n - ins);
}

uint_t bindingkey_hash(size_t idx, jl_value_t *data);
uint_t speccache_hash(size_t idx, jl_value_t *data);

static void jl_prune_idset(_Atomic(jl_svec_t*) *pkeys, _Atomic(jl_genericmemory_t*) *pkeyset, uint_t (*key_hash)(size_t, jl_value_t*), jl_value_t *parent) JL_GC_DISABLED
{
    jl_svec_t *keys = jl_atomic_load_relaxed(pkeys);
    size_t l = jl_svec_len(keys), i;
    if (l == 0)
        return;
    arraylist_t keys_list;
    arraylist_new(&keys_list, 0);
    for (i = 0; i < l; i++) {
        jl_value_t *k = jl_svecref(keys, i);
        if (k == jl_nothing)
            continue;
        if (ptrhash_get(&serialization_order, k) != HT_NOTFOUND)
            arraylist_push(&keys_list, k);
    }
    jl_genericmemory_t *keyset = jl_atomic_load_relaxed(pkeyset);
    _Atomic(jl_genericmemory_t*)keyset2;
    jl_atomic_store_relaxed(&keyset2, (jl_genericmemory_t*)jl_an_empty_memory_any);
    jl_svec_t *keys2 = jl_alloc_svec_uninit(keys_list.len);
    for (i = 0; i < keys_list.len; i++) {
        jl_binding_t *ref = (jl_binding_t*)keys_list.items[i];
        jl_svecset(keys2, i, ref);
        jl_smallintset_insert(&keyset2, parent, key_hash, i, (jl_value_t*)keys2);
    }
    void *idx = ptrhash_get(&serialization_order, keys);
    assert(idx != HT_NOTFOUND && idx != (void*)(uintptr_t)-1);
    assert(serialization_queue.items[(char*)idx - 1 - (char*)HT_NOTFOUND] == keys);
    ptrhash_put(&serialization_order, keys2, idx);
    serialization_queue.items[(char*)idx - 1 - (char*)HT_NOTFOUND] = keys2;

    idx = ptrhash_get(&serialization_order, keyset);
    assert(idx != HT_NOTFOUND && idx != (void*)(uintptr_t)-1);
    assert(serialization_queue.items[(char*)idx - 1 - (char*)HT_NOTFOUND] == keyset);
    ptrhash_put(&serialization_order, jl_atomic_load_relaxed(&keyset2), idx);
    serialization_queue.items[(char*)idx - 1 - (char*)HT_NOTFOUND] = jl_atomic_load_relaxed(&keyset2);
    jl_atomic_store_relaxed(pkeys, keys2);
    jl_gc_wb(parent, keys2);
    jl_atomic_store_relaxed(pkeyset, jl_atomic_load_relaxed(&keyset2));
    jl_gc_wb(parent, jl_atomic_load_relaxed(&keyset2));
}

static void jl_prune_method_specializations(jl_method_t *m) JL_GC_DISABLED
{
    jl_value_t *specializations_ = jl_atomic_load_relaxed(&m->specializations);
    if (!jl_is_svec(specializations_)) {
        if (ptrhash_get(&serialization_order, specializations_) == HT_NOTFOUND)
            record_field_change((jl_value_t **)&m->specializations, (jl_value_t*)jl_emptysvec);
        return;
    }
    jl_prune_idset((_Atomic(jl_svec_t*)*)&m->specializations, &m->speckeyset, speccache_hash, (jl_value_t*)m);
}

static void jl_prune_module_bindings(jl_module_t *m) JL_GC_DISABLED
{
    jl_prune_idset(&m->bindings, &m->bindingkeyset, bindingkey_hash, (jl_value_t*)m);
}

static void strip_slotnames(jl_array_t *slotnames, int n)
{
    // replace slot names with `?`, except unused_sym since the compiler looks at it
    jl_sym_t *questionsym = jl_symbol("?");
    int i;
    for (i = 0; i < n; i++) {
        jl_value_t *s = jl_array_ptr_ref(slotnames, i);
        if (s != (jl_value_t*)jl_unused_sym)
            jl_array_ptr_set(slotnames, i, questionsym);
    }
}

static jl_value_t *strip_codeinfo_meta(jl_method_t *m, jl_value_t *ci_, jl_code_instance_t *codeinst)
{
    jl_code_info_t *ci = NULL;
    JL_GC_PUSH1(&ci);
    int compressed = 0;
    if (!jl_is_code_info(ci_)) {
        compressed = 1;
        ci = jl_uncompress_ir(m, codeinst, (jl_value_t*)ci_);
    }
    else {
        ci = (jl_code_info_t*)ci_;
    }
    strip_slotnames(ci->slotnames, jl_array_len(ci->slotnames));
    ci->debuginfo = jl_nulldebuginfo;
    jl_gc_wb(ci, ci->debuginfo);
    jl_value_t *ret = (jl_value_t*)ci;
    if (compressed)
        ret = (jl_value_t*)jl_compress_ir(m, ci);
    JL_GC_POP();
    return ret;
}

static void strip_specializations_(jl_method_instance_t *mi)
{
    assert(jl_is_method_instance(mi));
    jl_code_instance_t *codeinst = jl_atomic_load_relaxed(&mi->cache);
    while (codeinst) {
        jl_value_t *inferred = jl_atomic_load_relaxed(&codeinst->inferred);
        if (inferred && inferred != jl_nothing && !jl_is_uint8(inferred)) {
            if (jl_options.strip_ir) {
                record_field_change((jl_value_t**)&codeinst->inferred, jl_nothing);
            }
            else if (jl_options.strip_metadata) {
                jl_value_t *stripped = strip_codeinfo_meta(mi->def.method, inferred, codeinst);
                if (jl_atomic_cmpswap_relaxed(&codeinst->inferred, &inferred, stripped)) {
                    jl_gc_wb(codeinst, stripped);
                }
            }
        }
        if (jl_options.strip_ir)
            record_field_change((jl_value_t**)&codeinst->edges, (jl_value_t*)jl_emptysvec);
        if (jl_options.strip_metadata)
            record_field_change((jl_value_t**)&codeinst->debuginfo, (jl_value_t*)jl_nulldebuginfo);
        codeinst = jl_atomic_load_relaxed(&codeinst->next);
    }
    if (jl_options.trim || jl_options.strip_ir) {
        record_field_change((jl_value_t**)&mi->backedges, NULL);
    }
}

static int strip_all_codeinfos__(jl_typemap_entry_t *def, void *_env)
{
    jl_method_t *m = def->func.method;
    if (m->source) {
        int stripped_ir = 0;
        if (jl_options.strip_ir) {
            int should_strip_ir = jl_options.trim;
            if (!should_strip_ir) {
                if (jl_atomic_load_relaxed(&m->unspecialized)) {
                    jl_code_instance_t *unspec = jl_atomic_load_relaxed(&jl_atomic_load_relaxed(&m->unspecialized)->cache);
                    if (unspec && jl_atomic_load_relaxed(&unspec->invoke)) {
                        // we have a generic compiled version, so can remove the IR
                        should_strip_ir = 1;
                    }
                }
            }
            if (!should_strip_ir) {
                int mod_setting = jl_get_module_compile(m->module);
                if (!(mod_setting == JL_OPTIONS_COMPILE_OFF || mod_setting == JL_OPTIONS_COMPILE_MIN)) {
                    // if the method is declared not to be compiled, keep IR for interpreter
                    should_strip_ir = 1;
                }
            }
            if (should_strip_ir) {
                record_field_change(&m->source, jl_nothing);
                record_field_change((jl_value_t**)&m->roots, NULL);
                stripped_ir = 1;
            }
        }
        if (jl_options.strip_metadata) {
            if (!stripped_ir) {
                m->source = strip_codeinfo_meta(m, m->source, NULL);
                jl_gc_wb(m, m->source);
            }
            jl_array_t *slotnames = jl_uncompress_argnames(m->slot_syms);
            JL_GC_PUSH1(&slotnames);
            int tostrip = jl_array_len(slotnames);
            // for keyword methods, strip only nargs to keep the keyword names at the end for reflection
            if (jl_tparam0(jl_unwrap_unionall(m->sig)) == (jl_value_t*)jl_kwcall_type)
                tostrip = m->nargs;
            strip_slotnames(slotnames, tostrip);
            m->slot_syms = jl_compress_argnames(slotnames);
            jl_gc_wb(m, m->slot_syms);
            JL_GC_POP();
        }
    }
    if (jl_options.strip_metadata) {
        record_field_change((jl_value_t**)&m->file, (jl_value_t*)jl_empty_sym);
        m->line = 0;
        record_field_change((jl_value_t**)&m->debuginfo, (jl_value_t*)jl_nulldebuginfo);
    }
    jl_value_t *specializations = jl_atomic_load_relaxed(&m->specializations);
    if (!jl_is_svec(specializations)) {
        strip_specializations_((jl_method_instance_t*)specializations);
    }
    else {
        size_t i, l = jl_svec_len(specializations);
        for (i = 0; i < l; i++) {
            jl_value_t *mi = jl_svecref(specializations, i);
            if (mi != jl_nothing)
                strip_specializations_((jl_method_instance_t*)mi);
        }
    }
    if (jl_atomic_load_relaxed(&m->unspecialized))
        strip_specializations_(jl_atomic_load_relaxed(&m->unspecialized));
    if (jl_options.strip_ir && m->root_blocks)
        record_field_change((jl_value_t**)&m->root_blocks, NULL);
    return 1;
}

static int strip_all_codeinfos_mt(jl_methtable_t *mt, void *_env)
{
    return jl_typemap_visitor(jl_atomic_load_relaxed(&mt->defs), strip_all_codeinfos__, NULL);
}

static void jl_strip_all_codeinfos(jl_array_t *mod_array)
{
    jl_foreach_reachable_mtable(strip_all_codeinfos_mt, mod_array, NULL);
}

static int strip_module(jl_module_t *m, jl_sym_t *docmeta_sym)
{
    size_t world = jl_atomic_load_relaxed(&jl_world_counter);
    jl_svec_t *table = jl_atomic_load_relaxed(&m->bindings);
    for (size_t i = 0; i < jl_svec_len(table); i++) {
        jl_binding_t *b = (jl_binding_t*)jl_svecref(table, i);
        if ((void*)b == jl_nothing)
            break;
        jl_sym_t *name = b->globalref->name;
        jl_value_t *v = jl_get_binding_value_in_world(b, world);
        if (v) {
            if (jl_is_module(v)) {
                jl_module_t *child = (jl_module_t*)v;
                if (child != m && child->parent == m && child->name == name) {
                    // this is the original/primary binding for the submodule
                    if (!strip_module(child, docmeta_sym))
                        return 0;
                }
            }
        }
        if (name == docmeta_sym) {
            if (jl_atomic_load_relaxed(&b->value))
                record_field_change((jl_value_t**)&b->value, jl_nothing);
            // TODO: this is a pretty stupidly unsound way to do this, but it is way to late here to do this correctly (by calling delete_binding and getting an updated world age then dropping all partitions from older worlds)
            jl_binding_partition_t *bp = jl_atomic_load_relaxed(&b->partitions);
            while (bp) {
                if (jl_bkind_is_defined_constant(jl_binding_kind(bp))) {
                    // XXX: bp->kind = PARTITION_KIND_UNDEF_CONST;
                    record_field_change((jl_value_t**)&bp->restriction, NULL);
                }
                bp = jl_atomic_load_relaxed(&bp->next);
            }
        }
    }
    return 1;
}


static void jl_strip_all_docmeta(jl_array_t *mod_array)
{
    jl_sym_t *docmeta_sym = NULL;
    if (jl_base_module) {
        jl_value_t *docs = jl_get_global(jl_base_module, jl_symbol("Docs"));
        if (docs && jl_is_module(docs)) {
            docmeta_sym = (jl_sym_t*)jl_get_global((jl_module_t*)docs, jl_symbol("META"));
        }
    }
    if (!docmeta_sym)
        return;
    for (size_t i = 0; i < jl_array_nrows(mod_array); i++) {
        jl_module_t *m = (jl_module_t*)jl_array_ptr_ref(mod_array, i);
        assert(jl_is_module(m));
        if (m->parent == m) // some toplevel modules (really just Base) aren't actually
            strip_module(m, docmeta_sym);
    }
}

// --- entry points ---

jl_genericmemory_t *jl_global_roots_list;
jl_genericmemory_t *jl_global_roots_keyset;
jl_mutex_t global_roots_lock;

jl_mutex_t precompile_field_replace_lock;
jl_svec_t *precompile_field_replace JL_GLOBALLY_ROOTED;

static inline jl_value_t *get_checked_fieldindex(const char *name, jl_datatype_t *st, jl_value_t *v, jl_value_t *arg, int mutabl)
{
    if (mutabl) {
        if (st == jl_module_type)
            jl_error("cannot assign variables in other modules");
        if (!st->name->mutabl)
            jl_errorf("%s: immutable struct of type %s cannot be changed", name, jl_symbol_name(st->name->name));
    }
    size_t idx;
    if (jl_is_long(arg)) {
        idx = jl_unbox_long(arg) - 1;
        if (idx >= jl_datatype_nfields(st))
            jl_bounds_error(v, arg);
    }
    else if (jl_is_symbol(arg)) {
        idx = jl_field_index(st, (jl_sym_t*)arg, 1);
        arg = jl_box_long(idx);
    }
    else {
        jl_value_t *ts[2] = {(jl_value_t*)jl_long_type, (jl_value_t*)jl_symbol_type};
        jl_value_t *t = jl_type_union(ts, 2);
        jl_type_error(name, t, arg);
    }
    if (mutabl && jl_field_isconst(st, idx)) {
        jl_errorf("%s: const field .%s of type %s cannot be changed", name,
                jl_symbol_name((jl_sym_t*)jl_svecref(jl_field_names(st), idx)), jl_symbol_name(st->name->name));
    }
    return arg;
}

JL_DLLEXPORT void jl_set_precompile_field_replace(jl_value_t *val, jl_value_t *field, jl_value_t *newval)
{
    if (!jl_generating_output())
        return;
    jl_datatype_t *st = (jl_datatype_t*)jl_typeof(val);
    jl_value_t *idx = get_checked_fieldindex("setfield!", st, val, field, 1);
    JL_GC_PUSH1(&idx);
    size_t idxval = jl_unbox_long(idx);
    jl_value_t *ft = jl_field_type_concrete(st, idxval);
    if (!jl_isa(newval, ft))
        jl_type_error("setfield!", ft, newval);
    JL_LOCK(&precompile_field_replace_lock);
    if (precompile_field_replace == NULL) {
        precompile_field_replace = jl_alloc_svec(3);
        jl_svecset(precompile_field_replace, 0, jl_alloc_vec_any(0));
        jl_svecset(precompile_field_replace, 1, jl_alloc_vec_any(0));
        jl_svecset(precompile_field_replace, 2, jl_alloc_vec_any(0));
    }
    jl_array_ptr_1d_push((jl_array_t*)jl_svecref(precompile_field_replace, 0), val);
    jl_array_ptr_1d_push((jl_array_t*)jl_svecref(precompile_field_replace, 1), idx);
    jl_array_ptr_1d_push((jl_array_t*)jl_svecref(precompile_field_replace, 2), newval);
    JL_GC_POP();
    JL_UNLOCK(&precompile_field_replace_lock);
}


JL_DLLEXPORT int jl_is_globally_rooted(jl_value_t *val JL_MAYBE_UNROOTED) JL_NOTSAFEPOINT
{
    if (jl_is_datatype(val)) {
        jl_datatype_t *dt = (jl_datatype_t*)val;
        if (jl_unwrap_unionall(dt->name->wrapper) == val)
            return 1;
        return (jl_is_tuple_type(val) ? dt->isconcretetype : !dt->hasfreetypevars); // aka is_cacheable from jltypes.c
    }
    if (jl_is_bool(val) || jl_is_symbol(val) ||
            val == (jl_value_t*)jl_any_type || val == (jl_value_t*)jl_bottom_type || val == (jl_value_t*)jl_core_module)
        return 1;
    if (val == ((jl_datatype_t*)jl_typeof(val))->instance)
        return 1;
    return 0;
}

static jl_value_t *extract_wrapper(jl_value_t *t JL_PROPAGATES_ROOT) JL_NOTSAFEPOINT JL_GLOBALLY_ROOTED
{
    t = jl_unwrap_unionall(t);
    if (jl_is_datatype(t))
        return ((jl_datatype_t*)t)->name->wrapper;
    return NULL;
}

JL_DLLEXPORT jl_value_t *jl_as_global_root(jl_value_t *val, int insert)
{
    if (jl_is_globally_rooted(val))
        return val;
    jl_value_t *tw = extract_wrapper(val);
    if (tw && (val == tw || jl_types_egal(val, tw)))
        return tw;
    if (jl_is_uint8(val))
        return jl_box_uint8(jl_unbox_uint8(val));
    if (jl_is_int32(val)) {
        int32_t n = jl_unbox_int32(val);
        if ((uint32_t)(n+512) < 1024)
            return jl_box_int32(n);
    }
    else if (jl_is_int64(val)) {
        uint64_t n = jl_unbox_uint64(val);
        if ((uint64_t)(n+512) < 1024)
            return jl_box_int64(n);
    }
    // check table before acquiring lock to reduce writer contention
    jl_value_t *rval = jl_idset_get(jl_global_roots_list, jl_global_roots_keyset, val);
    if (rval)
        return rval;
    JL_LOCK(&global_roots_lock);
    rval = jl_idset_get(jl_global_roots_list, jl_global_roots_keyset, val);
    if (rval) {
        val = rval;
    }
    else if (insert) {
        ssize_t idx;
        jl_global_roots_list = jl_idset_put_key(jl_global_roots_list, val, &idx);
        jl_global_roots_keyset = jl_idset_put_idx(jl_global_roots_list, jl_global_roots_keyset, idx);
    }
    else {
        val = NULL;
    }
    JL_UNLOCK(&global_roots_lock);
    return val;
}

// In addition to the system image (where `worklist = NULL`), this can also save incremental images with external linkage
static void jl_save_system_image_to_stream(ios_t *f, jl_array_t *mod_array,
                                           jl_array_t *module_init_order, jl_array_t *worklist, jl_array_t *extext_methods,
                                           jl_array_t *new_ext_cis, jl_query_cache *query_cache)
{
    htable_new(&field_replace, 0);
    htable_new(&bits_replace, 0);
    // strip metadata and IR when requested
    if (jl_options.strip_metadata || jl_options.strip_ir) {
        if (jl_options.strip_metadata) {
            jl_nulldebuginfo = (jl_debuginfo_t*)jl_get_global(jl_core_module, jl_symbol("NullDebugInfo"));
            if (jl_nulldebuginfo == NULL)
                jl_errorf("Core.NullDebugInfo required for --strip-metadata option");
        }
        jl_strip_all_codeinfos(mod_array);
        jl_strip_all_docmeta(mod_array);
    }
    // collect needed methods and replace method tables that are in the tags array
    htable_new(&new_methtables, 0);
    arraylist_t MIs;
    arraylist_new(&MIs, 0);
    arraylist_t gvars;
    arraylist_new(&gvars, 0);
    arraylist_t external_fns;
    arraylist_new(&external_fns, 0);
    // prepare hash table with any fields the user wanted us to rewrite during serialization
    if (precompile_field_replace) {
        jl_array_t *vals = (jl_array_t*)jl_svecref(precompile_field_replace, 0);
        jl_array_t *fields = (jl_array_t*)jl_svecref(precompile_field_replace, 1);
        jl_array_t *newvals = (jl_array_t*)jl_svecref(precompile_field_replace, 2);
        size_t i, l = jl_array_nrows(vals);
        assert(jl_array_nrows(fields) == l && jl_array_nrows(newvals) == l);
        for (i = 0; i < l; i++) {
            jl_value_t *val = jl_array_ptr_ref(vals, i);
            size_t field = jl_unbox_long(jl_array_ptr_ref(fields, i));
            jl_value_t *newval = jl_array_ptr_ref(newvals, i);
            jl_datatype_t *st = (jl_datatype_t*)jl_typeof(val);
            size_t offs = jl_field_offset(st, field);
            char *fldaddr = (char*)val + offs;
            if (jl_field_isptr(st, field)) {
                record_field_change((jl_value_t**)fldaddr, newval);
            }
            else if (jl_field_size(st, field) > 0) {
                // replace the bits
                ptrhash_put(&bits_replace, (void*)fldaddr, newval);
                // and any pointers inside
                jl_datatype_t *rty = (jl_datatype_t*)jl_typeof(newval);
                const jl_datatype_layout_t *layout = rty->layout;
                size_t j, np = layout->npointers;
                for (j = 0; j < np; j++) {
                    uint32_t ptr = jl_ptr_offset(rty, j);
                    record_field_change((jl_value_t**)fldaddr + ptr, *(((jl_value_t**)newval) + ptr));
                }
            }
        }
    }

    int en = jl_gc_enable(0);
    if (native_functions) {
        size_t num_gvars, num_external_fns;
        jl_get_llvm_gv_inits(native_functions, &num_gvars, NULL);
        arraylist_grow(&gvars, num_gvars);
        jl_get_llvm_gv_inits(native_functions, &num_gvars, gvars.items);
        jl_get_llvm_external_fns(native_functions, &num_external_fns, NULL);
        arraylist_grow(&external_fns, num_external_fns);
        jl_get_llvm_external_fns(native_functions, &num_external_fns,
                                 (jl_code_instance_t *)external_fns.items);
        if (jl_options.trim) {
            size_t num_mis;
            jl_get_llvm_cis(native_functions, &num_mis, NULL);
            arraylist_grow(&MIs, num_mis);

            // Record MethodInstances for user-provided code (as reported by codegen)
            jl_get_llvm_cis(native_functions, &num_mis, (jl_code_instance_t**)MIs.items);
            for (size_t i = 0; i < num_mis; i++) {
                jl_code_instance_t *ci = (jl_code_instance_t*)MIs.items[i];
                MIs.items[i] = (void*)jl_get_ci_mi(ci);
            }

            // Record MethodInstances for built-ins (used when dynamically dispatching to a
            // built-in, e.g., in the Core._apply_iterate implementation)
            jl_datatype_t *tt = NULL;
            JL_GC_PUSH1(&tt);
            for (size_t i = 0; i < jl_n_builtins; i++) {
                jl_value_t *builtin = jl_builtin_instances[i];
                if (builtin == NULL)
                    continue;

                jl_datatype_t *dt = (jl_datatype_t*)jl_typeof(builtin);
                jl_value_t *params[2];
                params[0] = dt->name->wrapper;
                params[1] = jl_tparam0(jl_anytuple_type);
                tt = (jl_datatype_t*)jl_apply_tuple_type_v(params, 2);
                jl_method_instance_t *mi = (jl_method_instance_t *)jl_method_lookup_by_tt(
                    tt, /* world */ 1, /* mt */ jl_nothing
                );
                assert(!jl_is_nothing(mi));
                arraylist_push(&MIs, mi);
            }
            JL_GC_POP();
        }
    }
    if (jl_options.trim) {
        jl_rebuild_methtables(&MIs, &new_methtables);
    }

    nsym_tag = 0;
    htable_new(&symbol_table, 0);
    htable_new(&fptr_to_id, jl_n_builtins);
    uintptr_t i;
    for (i = 0; i < jl_n_builtins; i++) {
        ptrhash_put(&fptr_to_id, (void*)(uintptr_t)jl_builtin_f_addrs[i], (void*)(i + 2));
    }
    htable_new(&serialization_order, 25000);
    htable_new(&nullptrs, 0);
    arraylist_new(&object_worklist, 0);
    arraylist_new(&serialization_queue, 0);
    ios_t sysimg, const_data, symbols, relocs, gvar_record, fptr_record;
    ios_mem(&sysimg, 0);
    ios_mem(&const_data, 0);
    ios_mem(&symbols, 0);
    ios_mem(&relocs, 0);
    ios_mem(&gvar_record, 0);
    ios_mem(&fptr_record, 0);
    jl_serializer_state s = {0};
    s.query_cache = query_cache;
    s.incremental = !(worklist == NULL);
    s.s = &sysimg;
    s.const_data = &const_data;
    s.symbols = &symbols;
    s.relocs = &relocs;
    s.gvar_record = &gvar_record;
    s.fptr_record = &fptr_record;
    s.ptls = jl_current_task->ptls;
    arraylist_new(&s.memowner_list, 0);
    arraylist_new(&s.memref_list, 0);
    arraylist_new(&s.relocs_list, 0);
    arraylist_new(&s.gctags_list, 0);
    arraylist_new(&s.uniquing_types, 0);
    arraylist_new(&s.uniquing_super, 0);
    arraylist_new(&s.uniquing_objs, 0);
    arraylist_new(&s.fixup_types, 0);
    arraylist_new(&s.fixup_objs, 0);
    s.buildid_depmods_idxs = image_to_depmodidx(mod_array);
    s.link_ids_relocs = jl_alloc_array_1d(jl_array_int32_type, 0);
    s.link_ids_gctags = jl_alloc_array_1d(jl_array_int32_type, 0);
    s.link_ids_gvars = jl_alloc_array_1d(jl_array_int32_type, 0);
    s.link_ids_external_fnvars = jl_alloc_array_1d(jl_array_int32_type, 0);
    s.method_roots_list = NULL;
    htable_new(&s.method_roots_index, 0);
    arraylist_new(&s.import_objs, 0);
    arraylist_new(&s.import_deps, 0);
    htable_new(&s.import_index, 0);
    jl_value_t **_tags[NUM_TAGS];
    jl_value_t ***tags = s.incremental ? NULL : _tags;
    if (worklist) {
        s.method_roots_list = jl_alloc_vec_any(0);
        s.worklist_key = jl_worklist_key(worklist);
    }
    else {
        get_tags(_tags);
    }

    if (worklist == NULL) {
        // empty!(Core.ARGS)
        if (jl_core_module != NULL) {
            jl_array_t *args = (jl_array_t*)jl_get_global(jl_core_module, jl_symbol("ARGS"));
            if (args != NULL) {
                jl_array_del_end(args, jl_array_len(args));
            }
        }
    }
    jl_bigint_type = jl_base_module ? jl_get_global(jl_base_module, jl_symbol("BigInt")) : NULL;
    if (jl_bigint_type) {
        gmp_limb_size = jl_unbox_long(jl_get_global((jl_module_t*)jl_get_global(jl_base_module, jl_symbol("GMP")),
                                                    jl_symbol("BITS_PER_LIMB"))) / 8;
    }
    jl_genericmemory_t *global_roots_list = NULL;
    jl_genericmemory_t *global_roots_keyset = NULL;

    { // step 1: record values (recursively) that need to go in the image
        size_t i;
        if (worklist == NULL) {
            for (i = 0; tags[i] != NULL; i++) {
                jl_value_t *tag = *tags[i];
                jl_queue_for_serialization(&s, tag);
            }
            for (i = 0; i < jl_n_builtins; i++)
                jl_queue_for_serialization(&s, jl_builtin_instances[i]);
#define XX(name, type) jl_queue_for_serialization(&s, (jl_value_t*)jl_##name);
            JL_EXPORTED_DATA_POINTERS(XX)
#undef XX
#define XX(name, type) jl_queue_for_serialization(&s, (jl_value_t*)jl_##name);
            JL_CONST_GLOBAL_VARS(XX)
#undef XX
            jl_queue_for_serialization(&s, s.ptls->root_task->tls);
        }
        else {
            // Queue the worklist itself as the first item we serialize
            jl_queue_for_serialization(&s, worklist);
            jl_queue_for_serialization(&s, module_init_order);
        }
        // step 1.1: as needed, serialize the data needed for insertion into the running system
        if (extext_methods) {
            // Queue method extensions
            jl_queue_for_serialization(&s, extext_methods);
            // Queue the new specializations
            jl_queue_for_serialization(&s, new_ext_cis);
        }
        jl_serialize_reachable(&s);
        // step 1.2: ensure all gvars are part of the sysimage too
        record_gvars(&s, &gvars);
        record_external_fns(&s, &external_fns);
        if (jl_options.trim)
            record_gvars(&s, &MIs);
        jl_serialize_reachable(&s);
        // Beyond this point, all content should already have been visited, so now we can prune
        // the rest and add some internal root arrays.
        // step 1.3: include some other special roots
        if (s.incremental) {
            // Queue the new roots array
            jl_queue_for_serialization(&s, s.method_roots_list);
            jl_serialize_reachable(&s);
        }
        // step 1.4: prune (garbage collect) special weak references from the jl_global_roots_list
        if (worklist == NULL) {
            global_roots_list = jl_alloc_memory_any(0);
            global_roots_keyset = jl_alloc_memory_any(0);
            for (size_t i = 0; i < jl_global_roots_list->length; i++) {
                jl_value_t *val = jl_genericmemory_ptr_ref(jl_global_roots_list, i);
                if (val && ptrhash_get(&serialization_order, val) != HT_NOTFOUND) {
                    ssize_t idx;
                    global_roots_list = jl_idset_put_key(global_roots_list, val, &idx);
                    global_roots_keyset = jl_idset_put_idx(global_roots_list, global_roots_keyset, idx);
                }
            }
            jl_queue_for_serialization(&s, global_roots_list);
            jl_queue_for_serialization(&s, global_roots_keyset);
            jl_serialize_reachable(&s);
        }
        // step 1.5: prune (garbage collect) some special weak references known caches
        for (i = 0; i < serialization_queue.len; i++) {
            jl_value_t *v = (jl_value_t*)serialization_queue.items[i];
            if (jl_is_method(v)) {
                if (jl_options.trim)
                    jl_prune_method_specializations((jl_method_t*)v);
            }
            else if (jl_is_module(v)) {
                if (jl_options.trim)
                    jl_prune_module_bindings((jl_module_t*)v);
            }
            else if (jl_is_typename(v)) {
                jl_typename_t *tn = (jl_typename_t*)v;
                jl_atomic_store_relaxed(&tn->cache,
                    jl_prune_type_cache_hash(jl_atomic_load_relaxed(&tn->cache)));
                jl_gc_wb(tn, jl_atomic_load_relaxed(&tn->cache));
                jl_prune_type_cache_linear(jl_atomic_load_relaxed(&tn->linearcache));
            }
            else if (jl_is_method_instance(v)) {
                jl_method_instance_t *mi = (jl_method_instance_t*)v;
                jl_value_t *backedges = get_replaceable_field((jl_value_t**)&mi->backedges, 1);
                jl_prune_mi_backedges((jl_array_t*)backedges);
            }
            else if (jl_is_binding(v)) {
                jl_binding_t *b = (jl_binding_t*)v;
                jl_value_t *backedges = get_replaceable_field((jl_value_t**)&b->backedges, 1);
                jl_prune_binding_backedges((jl_array_t*)backedges);
            }
            else if (jl_is_mtable(v)) {
                jl_methtable_t *mt = (jl_methtable_t*)v;
                jl_value_t *backedges = get_replaceable_field((jl_value_t**)&mt->backedges, 1);
                jl_prune_mt_backedges((jl_genericmemory_t*)backedges);
            }
        }
    }

    uint32_t external_fns_begin = 0;
    { // step 2: build all the sysimg sections
        write_padding(&sysimg, sizeof(uintptr_t));
        jl_write_values(&s);
        external_fns_begin = write_gvars(&s, &gvars, &external_fns);
    }

    // This ensures that we can use the low bit of addresses for
    // identifying end pointers in gc's eytzinger search.
    write_padding(&sysimg, 4 - (sysimg.size % 4));
    write_padding(&const_data, 4 - (const_data.size % 4));

    if (sysimg.size > ((uintptr_t)1 << RELOC_TAG_OFFSET)) {
        jl_printf(
            JL_STDERR,
            "ERROR: system image too large: sysimg.size is 0x%" PRIxPTR " but the limit is 0x%" PRIxPTR "\n",
            (uintptr_t)sysimg.size,
            ((uintptr_t)1 << RELOC_TAG_OFFSET)
        );
        jl_exit(1);
    }
    if (const_data.size / sizeof(void*) > ((uintptr_t)1 << RELOC_TAG_OFFSET)) {
        jl_printf(
            JL_STDERR,
            "ERROR: system image too large: const_data.size is 0x%" PRIxPTR " but the limit is 0x%" PRIxPTR "\n",
            (uintptr_t)const_data.size,
            ((uintptr_t)1 << RELOC_TAG_OFFSET)*sizeof(void*)
        );
        jl_exit(1);
    }

    // step 3: combine all of the sections into one file
    assert(ios_pos(f) % JL_CACHE_BYTE_ALIGNMENT == 0);
    ssize_t sysimg_offset = ios_pos(f);
    write_uint(f, sysimg.size - sizeof(uintptr_t));
    ios_seek(&sysimg, sizeof(uintptr_t));
    ios_copyall(f, &sysimg);
    size_t sysimg_size = s.s->size;
    assert(ios_pos(f) - sysimg_offset == sysimg_size);
    ios_close(&sysimg);

    write_uint(f, const_data.size);
    // realign stream to max-alignment for data
    write_padding(f, LLT_ALIGN(ios_pos(f), JL_CACHE_BYTE_ALIGNMENT) - ios_pos(f));
    ios_seek(&const_data, 0);
    ios_copyall(f, &const_data);
    ios_close(&const_data);

    write_uint(f, symbols.size);
    write_padding(f, LLT_ALIGN(ios_pos(f), 8) - ios_pos(f));
    ios_seek(&symbols, 0);
    ios_copyall(f, &symbols);
    ios_close(&symbols);

    // Prepare and write the relocations sections, now that the rest of the image is laid out
    char *base = &f->buf[0];
    jl_finish_relocs(base + sysimg_offset, sysimg_size, &s.gctags_list);
    jl_finish_relocs(base + sysimg_offset, sysimg_size, &s.relocs_list);
    jl_write_offsetlist(s.relocs, sysimg_size, &s.gctags_list);
    jl_write_offsetlist(s.relocs, sysimg_size, &s.relocs_list);
    jl_write_offsetlist(s.relocs, sysimg_size, &s.memowner_list);
    jl_write_offsetlist(s.relocs, sysimg_size, &s.memref_list);
    if (s.incremental) {
        jl_write_arraylist(s.relocs, &s.uniquing_types);
        jl_write_arraylist(s.relocs, &s.uniquing_objs);
        jl_write_arraylist(s.relocs, &s.fixup_types);
    }
    jl_write_arraylist(s.relocs, &s.fixup_objs);
    write_uint(f, relocs.size);
    write_padding(f, LLT_ALIGN(ios_pos(f), 8) - ios_pos(f));
    ios_seek(&relocs, 0);
    ios_copyall(f, &relocs);
    ios_close(&relocs);

    write_uint(f, gvar_record.size);
    write_padding(f, LLT_ALIGN(ios_pos(f), 8) - ios_pos(f));
    ios_seek(&gvar_record, 0);
    ios_copyall(f, &gvar_record);
    ios_close(&gvar_record);

    write_uint(f, fptr_record.size);
    write_padding(f, LLT_ALIGN(ios_pos(f), 8) - ios_pos(f));
    ios_seek(&fptr_record, 0);
    ios_copyall(f, &fptr_record);
    ios_close(&fptr_record);

    { // step 4: record locations of special roots
        write_padding(f, LLT_ALIGN(ios_pos(f), 8) - ios_pos(f));
        s.s = f;
        if (worklist == NULL) {
            size_t i;
            for (i = 0; tags[i] != NULL; i++) {
                jl_value_t *tag = *tags[i];
                jl_write_value(&s, tag);
            }
            for (i = 0; i < jl_n_builtins; i++)
                jl_write_value(&s, jl_builtin_instances[i]);
#define XX(name, type) jl_write_value(&s, (jl_value_t*)jl_##name);
            JL_EXPORTED_DATA_POINTERS(XX)
#undef XX
#define XX(name, type) jl_write_value(&s, (jl_value_t*)jl_##name);
            JL_CONST_GLOBAL_VARS(XX)
#undef XX
            jl_write_value(&s, global_roots_list);
            jl_write_value(&s, global_roots_keyset);
            jl_write_value(&s, s.ptls->root_task->tls);
            write_uint32(f, jl_get_gs_ctr());
            size_t world = jl_atomic_load_acquire(&jl_world_counter);
            // assert(world == precompilation_world); // This triggers on a normal build of julia
            write_uint(f, world);
            write_uint(f, jl_typeinf_world);
        }
        else {
            jl_write_value(&s, worklist);
            // save module initialization order
            size_t i, l = jl_array_len(module_init_order);
            for (i = 0; i < l; i++) {
                // verify that all these modules were saved
                assert(ptrhash_get(&serialization_order, jl_array_ptr_ref(module_init_order, i)) != HT_NOTFOUND);
            }
            jl_write_value(&s, module_init_order);
            jl_write_value(&s, extext_methods);
            jl_write_value(&s, new_ext_cis);
            jl_write_value(&s, s.method_roots_list);
        }
        write_uint32(f, jl_array_len(s.link_ids_gctags));
        ios_write(f, (char*)jl_array_data(s.link_ids_gctags, uint32_t), jl_array_len(s.link_ids_gctags) * sizeof(uint32_t));
        write_uint32(f, jl_array_len(s.link_ids_relocs));
        ios_write(f, (char*)jl_array_data(s.link_ids_relocs, uint32_t), jl_array_len(s.link_ids_relocs) * sizeof(uint32_t));
        write_uint32(f, jl_array_len(s.link_ids_gvars));
        ios_write(f, (char*)jl_array_data(s.link_ids_gvars, uint32_t), jl_array_len(s.link_ids_gvars) * sizeof(uint32_t));
        write_uint32(f, jl_array_len(s.link_ids_external_fnvars));
        ios_write(f, (char*)jl_array_data(s.link_ids_external_fnvars, uint32_t), jl_array_len(s.link_ids_external_fnvars) * sizeof(uint32_t));
        write_uint32(f, external_fns_begin);
        extkey_build_paths(mod_array);
        extkey_build_tvars(&s);
        jl_write_import_table(&s, f);
    }

    if (getenv("JULIA_IMPORT_KEYS"))
        jl_report_import_keys(&s);
    if (getenv("JULIA_SHADOW_RESOLVE"))
        jl_shadow_resolve_imports(&s, mod_array);
    if (getenv("JULIA_IDHASH_TAINT"))
        jl_report_idhash_taint(&s);
    if (getenv("JULIA_KEY_PARSE"))
        jl_check_key_parse(&s, mod_array);

    assert(object_worklist.len == 0);
    arraylist_free(&object_worklist);
    arraylist_free(&serialization_queue);
    arraylist_free(&layout_table);
    arraylist_free(&s.uniquing_types);
    arraylist_free(&s.uniquing_super);
    arraylist_free(&s.uniquing_objs);
    arraylist_free(&s.fixup_types);
    arraylist_free(&s.fixup_objs);
    arraylist_free(&s.memowner_list);
    arraylist_free(&s.memref_list);
    arraylist_free(&s.relocs_list);
    arraylist_free(&s.gctags_list);
    arraylist_free(&gvars);
    arraylist_free(&external_fns);
    htable_free(&s.method_roots_index);
    arraylist_free(&s.import_objs);
    arraylist_free(&s.import_deps);
    htable_free(&s.import_index);
    htable_free(&field_replace);
    htable_free(&bits_replace);
    htable_free(&serialization_order);
    htable_free(&nullptrs);
    htable_free(&symbol_table);
    htable_free(&fptr_to_id);
    htable_free(&new_methtables);
    nsym_tag = 0;

    jl_gc_enable(en);
}

static int ci_not_internal_cache(jl_code_instance_t *ci)
{
    jl_method_instance_t *mi = jl_get_ci_mi(ci);
    return !(jl_atomic_load_relaxed(&ci->flags) & JL_CI_FLAGS_NATIVE_CACHE_VALID) || jl_object_in_image(mi->def.value);
}

static void jl_write_header_for_incremental(ios_t *f, jl_array_t *worklist, jl_array_t *mod_array, jl_array_t **udeps, int64_t *srctextpos, int64_t *checksumpos)
{
    *checksumpos = write_header(f, 0);
    write_uint8(f, jl_cache_flags());
    // write description of contents (name, uuid, buildid)
    write_worklist_for_header(f, worklist);
    // Determine unique (module, abspath, fsize, hash, mtime) dependencies for the files defining modules in the worklist
    // (see Base._require_dependencies). These get stored in `udeps` and written to the ji-file header
    // (abspath will be converted to a relocateable @depot path before writing, cf. Base.replace_depot_path).
    // Also write Preferences.
    // last word of the dependency list is the end of the data / start of the srctextpos
    *srctextpos = write_dependency_list(f, worklist, udeps);  // srctextpos: position of srctext entry in header index (update later)
    // write description of requirements for loading (modules that must be pre-loaded if initialization is to succeed)
    // this can return errors during deserialize,
    // best to keep it early (before any actual initialization)
    write_mod_list(f, mod_array);
}

JL_DLLEXPORT void jl_create_system_image(void **_native_data, jl_array_t *worklist, bool_t emit_split,
                                         ios_t **s, ios_t **z, jl_array_t **udeps, int64_t *srctextpos, jl_array_t *module_init_order)
{
    if (jl_options.strip_ir || jl_options.trim) {
        // make sure this is precompiled for jl_foreach_reachable_mtable
        jl_get_loaded_modules();
    }
    jl_gc_collect(JL_GC_FULL);
    jl_gc_collect(JL_GC_INCREMENTAL);   // sweep finalizers
    JL_TIMING(SYSIMG_DUMP, SYSIMG_DUMP);

    // iff emit_split
    // write header and src_text to one file f/s
    // write systemimg to a second file ff/z
    jl_task_t *ct = jl_current_task;
    ios_t *f = (ios_t*)malloc_s(sizeof(ios_t));
    ios_mem(f, 0);

    ios_t *ff = NULL;
    if (emit_split) {
        ff = (ios_t*)malloc_s(sizeof(ios_t));
        ios_mem(ff, 0);
    } else {
        ff = f;
    }

    jl_array_t *mod_array = NULL, *extext_methods = NULL, *new_ext_cis = NULL, *ext_foreign_cis = NULL;
    int64_t checksumpos = 0;
    int64_t checksumpos_ff = 0;
    int64_t datastartpos = 0;
    JL_GC_PUSH4(&mod_array, &extext_methods, &new_ext_cis, &ext_foreign_cis);

    ext_foreign_cis = jl_alloc_vec_any(0);

    mod_array = jl_get_loaded_modules();  // __toplevel__ modules loaded in this session (from Base.loaded_modules_array)
    if (worklist) {
        if (_native_data != NULL) {
            if (suppress_precompile)
                newly_inferred = NULL;
            *_native_data = jl_create_native(NULL, 0, 1, jl_atomic_load_acquire(&jl_world_counter), NULL, suppress_precompile ? (jl_array_t*)jl_an_empty_vec_any : worklist, 0, module_init_order, ext_foreign_cis);
        }
        jl_write_header_for_incremental(f, worklist, mod_array, udeps, srctextpos, &checksumpos);
        if (emit_split) {
            checksumpos_ff = write_header(ff, 1);
            write_uint8(ff, jl_cache_flags());
            write_mod_list(ff, mod_array);
        }
        else {
            checksumpos_ff = checksumpos;
        }
    }
    else if (_native_data != NULL) {
        *_native_data = jl_create_native(NULL, jl_options.trim, 0, jl_atomic_load_acquire(&jl_world_counter), mod_array, NULL, jl_options.compile_enabled == JL_OPTIONS_COMPILE_ALL, module_init_order, ext_foreign_cis);
    }
    if (_native_data != NULL)
        native_functions = *_native_data;

    // Make sure we don't run any Julia code concurrently after this point
    // since it will invalidate our serialization preparations
    jl_gc_enable_finalizers(ct, 0);
    assert((ct->reentrant_timing & 0b1110) == 0);
    ct->reentrant_timing |= 0b1000;
    if (worklist) {
        // extext_methods: [method1, ...], worklist-owned "extending external" methods added to functions owned by modules outside the worklist

        // Save the inferred code from newly inferred, external methods
        if (native_functions) {
            arraylist_t CIs;
            arraylist_new(&CIs, 0);
            size_t num_cis;
            jl_get_llvm_cis(native_functions, &num_cis, NULL);
            arraylist_grow(&CIs, num_cis);
            jl_get_llvm_cis(native_functions, &num_cis, (jl_code_instance_t**)CIs.items);
            // Create a filtered list of the compiled code instances that are
            // possibly not referenced via any other way but valid for the
            // Method cache field of an external method
            new_ext_cis = jl_alloc_vec_any(0);
            for (size_t i = 0; i < num_cis; i++) {
                jl_code_instance_t *ci = (jl_code_instance_t*)CIs.items[i];
                if (ci_not_internal_cache(ci))
                    jl_array_ptr_1d_push(new_ext_cis, (jl_value_t*)ci);
            }
            arraylist_free(&CIs);
        }
        else {
            new_ext_cis = jl_compute_new_ext_cis();
        }

        // Merge foreign & external CIs
        if (ext_foreign_cis) {
            size_t n_ext = jl_array_nrows(ext_foreign_cis);
            for (size_t i = 0; i < n_ext; i++) {
                jl_array_ptr_1d_push(new_ext_cis, jl_array_ptr_ref(ext_foreign_cis, i));
            }
        }
        ext_foreign_cis = NULL; // not needed anymore, free it

        // Collect method extensions
        extext_methods = jl_alloc_vec_any(0);
        jl_collect_extext_methods(extext_methods, mod_array);

        if (!emit_split) {
            write_int32(f, 0); // No clone_targets
            write_padding(f, LLT_ALIGN(ios_pos(f), JL_CACHE_BYTE_ALIGNMENT) - ios_pos(f));
        }
        else {
            write_padding(ff, LLT_ALIGN(ios_pos(ff), JL_CACHE_BYTE_ALIGNMENT) - ios_pos(ff));
        }
        datastartpos = ios_pos(ff);
    }

    jl_query_cache query_cache;
    init_query_cache(&query_cache);
    jl_save_system_image_to_stream(ff, mod_array, module_init_order, worklist, extext_methods, new_ext_cis, &query_cache);
    if (_native_data != NULL)
        native_functions = NULL;
    // make sure we don't run any Julia code concurrently before this point
    // Re-enable running julia code for postoutput hooks, atexit, etc.
    jl_gc_enable_finalizers(ct, 1);
    ct->reentrant_timing &= ~0b1000u;

    if (worklist) {
        // Go back and update the checksum in the header
        int64_t dataendpos = ios_pos(ff);
        uint32_t checksum = jl_crc32c(0, &ff->buf[datastartpos], dataendpos - datastartpos);
        ios_seek(ff, checksumpos_ff);
        write_uint64(ff, checksum | ((uint64_t)0xfafbfcfd << 32));
        write_uint64(ff, datastartpos);
        write_uint64(ff, dataendpos);
        ios_seek(ff, dataendpos);

        // Write the checksum to the split header if necessary
        if (emit_split) {
            int64_t cur = ios_pos(f);
            ios_seek(f, checksumpos);
            write_uint64(f, checksum | ((uint64_t)0xfafbfcfd << 32));
            ios_seek(f, cur);
            // Next we will write the clone_targets and afterwards the srctext
        }
    }

    destroy_query_cache(&query_cache);

    JL_GC_POP();
    *s = f;
    if (emit_split)
        *z = ff;
    return;
}

// Takes in a path of the form "usr/lib/julia/sys.so"
JL_DLLEXPORT jl_image_buf_t jl_preload_sysimg(const char *fname)
{
    if (jl_sysimage_buf.kind != JL_IMAGE_KIND_NONE)
        return jl_sysimage_buf;

    char *dot = (char*) strrchr(fname, '.');
    int is_ji = (dot && !strcmp(dot, ".ji"));

    if (is_ji) {
        // .ji extension => load .ji file only
        ios_t f;

        if (ios_file(&f, fname, 1, 0, 0, 0) == NULL)
            jl_errorf("System image file \"%s\" not found.", fname);
        ios_bufmode(&f, bm_none);

        ios_seek_end(&f);
        size_t len = ios_pos(&f);
        char *sysimg = (char*)jl_gc_perm_alloc(len, 0, 64, 0);
        ios_seek(&f, 0);

        if (ios_readall(&f, sysimg, len) != len)
            jl_errorf("Error reading system image file.");

        ios_close(&f);

        jl_sysimage_buf = (jl_image_buf_t) {
            .kind = JL_IMAGE_KIND_JI,
            .pointers = NULL,
            .data = sysimg,
            .size = len,
            .base = 0,
        };
        return jl_sysimage_buf;
    } else {
        // Get handle to sys.so
        return jl_set_sysimg_so(jl_load_dynamic_library(fname, JL_RTLD_LOCAL | JL_RTLD_NOW, 1));
    }
}


static void jl_prefetch_system_image(const char *data, size_t size)
{
    size_t page_size = jl_getpagesize(); /* jl_page_size is not set yet when loading sysimg */
    void *start = (void *)((uintptr_t)data & ~(page_size - 1));
    size_t size_aligned = LLT_ALIGN(size, page_size);
#ifdef _OS_WINDOWS_
    WIN32_MEMORY_RANGE_ENTRY entry = {start, size_aligned};
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
#else
    madvise(start, size_aligned, MADV_WILLNEED);
#endif
}

JL_DLLEXPORT void jl_image_unpack_uncomp(void *handle, jl_image_buf_t *image)
{
    size_t *plen;
    uint32_t *pchecksum;
    jl_dlsym(handle, "jl_system_image_size", (void **)&plen, 1, 0);
    jl_dlsym(handle, "jl_system_image_data", (void **)&image->data, 1, 0);
    jl_dlsym(handle, "jl_image_pointers", (void**)&image->pointers, 1, 0);
    jl_dlsym(handle, "jl_system_image_checksum", (void **)&pchecksum, 1, 0);
    image->size = *plen;
    image->checksum = *pchecksum;
    jl_prefetch_system_image(image->data, image->size);
}

JL_DLLEXPORT void jl_image_unpack_zstd(void *handle, jl_image_buf_t *image)
{
    size_t *plen;
    uint32_t *pchecksum;
    const char *data;
    jl_dlsym(handle, "jl_system_image_size", (void **)&plen, 1, 0);
    jl_dlsym(handle, "jl_system_image_data", (void **)&data, 1, 0);
    jl_dlsym(handle, "jl_image_pointers", (void **)&image->pointers, 1, 0);
    jl_dlsym(handle, "jl_system_image_checksum", (void **)&pchecksum, 1, 0);
    image->checksum = *pchecksum;
    jl_prefetch_system_image(data, *plen);
    image->size = ZSTD_getFrameContentSize(data, *plen);
    size_t page_size = jl_getpagesize(); /* jl_page_size is not set yet when loading sysimg */
    size_t aligned_size = LLT_ALIGN(image->size, page_size);
    int fail = 0;
#if defined(_OS_WINDOWS_)
    size_t large_page_size = GetLargePageMinimum();
    image->data = NULL;
    if (large_page_size > 0 && image->size > 4 * large_page_size) {
        size_t aligned_size = LLT_ALIGN(image->size, large_page_size);
        image->data = (char *)VirtualAlloc(
            NULL, aligned_size, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
    }
    if (!image->data) {
        /* Try small pages if large pages failed. */
        image->data = (char *)VirtualAlloc(NULL, aligned_size, MEM_COMMIT | MEM_RESERVE,
                                           PAGE_READWRITE);
    }
    fail = !image->data;
#else
    image->data = (char *)mmap(NULL, aligned_size, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    fail = image->data == (void *)-1;
#endif
    if (fail) {
        const char *err;
#if defined(_OS_WINDOWS_)
        char err_buf[256];
        win32_formatmessage(GetLastError(), err_buf, sizeof(err_buf));
        err = err_buf;
#else
        err = strerror(errno);
#endif
        jl_printf(JL_STDERR, "ERROR: failed to allocate memory for system image: %s\n",
                  err);
        jl_exit(1);
    }

    ZSTD_decompress((void *)image->data, image->size, data, *plen);
    size_t len = (*plen) & ~(page_size - 1);
#ifdef _OS_WINDOWS_
    if (len)
        VirtualFree((void *)data, len, MEM_RELEASE);
#else
    munmap((void *)data, len);
#endif
}

// From a shared library handle, verify consistency and return a jl_image_buf_t
static jl_image_buf_t get_image_buf(void *handle, int is_pkgimage)
{
    // verify that the linker resolved the symbols in this image against ourselves (libjulia-internal)
    void** (*get_jl_RTLD_DEFAULT_handle_addr)(void) = NULL;
    if (handle != jl_RTLD_DEFAULT_handle) {
        int symbol_found = jl_dlsym(handle, "get_jl_RTLD_DEFAULT_handle_addr", (void **)&get_jl_RTLD_DEFAULT_handle_addr, 0, 0);
        if (!symbol_found || (void*)&jl_RTLD_DEFAULT_handle != (get_jl_RTLD_DEFAULT_handle_addr()))
            jl_error("Image file failed consistency check: maybe opened the wrong version?");
    }

    jl_image_unpack_func_t **unpack;
    jl_image_buf_t image = {
        .kind = JL_IMAGE_KIND_SO,
        .pointers = NULL,
        .data = NULL,
        .size = 0,
        .base = 0,
    };

    // verification passed, lookup the buffer pointers
    if (jl_image_unpack == NULL || is_pkgimage) {
        // in the usual case, the sysimage was not statically linked to libjulia-internal
        // look up the external sysimage symbols via the dynamic linker
        jl_dlsym(handle, "jl_image_unpack", (void **)&unpack, 1, 0);
    }
    else {
        // the sysimage was statically linked directly against libjulia-internal
        // use the internal symbols
        unpack = &jl_image_unpack;
    }
    (*unpack)(handle, &image);

#ifdef _OS_WINDOWS_
    image.base = (intptr_t)handle;
#else
    Dl_info dlinfo;
    if (dladdr((void*)image.pointers, &dlinfo) != 0)
        image.base = (intptr_t)dlinfo.dli_fbase;
    else
        image.base = 0;
#endif

    return image;
}

// Allow passing in a module handle directly, rather than a path
JL_DLLEXPORT jl_image_buf_t jl_set_sysimg_so(void *handle)
{
    if (jl_sysimage_buf.kind != JL_IMAGE_KIND_NONE)
        return jl_sysimage_buf;

    jl_sysimage_buf = get_image_buf(handle, /* is_pkgimage */ 0);
    return jl_sysimage_buf;
}

#ifndef JL_NDEBUG
// skip the performance optimizations of jl_types_equal and just use subtyping directly
// one of these types is invalid - that's why we're doing the recache type operation
// static int jl_invalid_types_equal(jl_datatype_t *a, jl_datatype_t *b)
// {
//     return jl_subtype((jl_value_t*)a, (jl_value_t*)b) && jl_subtype((jl_value_t*)b, (jl_value_t*)a);
// }
#endif

extern void rebuild_image_blob_tree(void);
extern void export_jl_small_typeof(void);
extern void export_jl_sysimg_globals(void);

// When an image is loaded with ignore_native, all subsequent image loads must ignore
// native code in the cache-file since we can't gurantuee that there are no call edges
// into the native code of the image. See https://github.com/JuliaLang/julia/pull/52123#issuecomment-1959965395.
int IMAGE_NATIVE_CODE_TAINTED = 0;

// TODO: This should possibly be in Julia
static int jl_validate_binding_partition(jl_binding_t *b, jl_binding_partition_t *bpart, size_t mod_idx, int unchanged_implicit, int no_replacement)
{
    if (jl_atomic_load_relaxed(&bpart->max_world) != ~(size_t)0)
        return 1;
    size_t raw_kind = bpart->kind;
    enum jl_partition_kind kind = (enum jl_partition_kind)(raw_kind & PARTITION_MASK_KIND);
    if (!unchanged_implicit && jl_bkind_is_some_implicit(kind)) {
        // TODO: Should we actually update this in place or delete it from the partitions list
        // and allocate a fresh bpart?
        jl_update_loaded_bpart(b, bpart);
        bpart->kind |= (raw_kind & PARTITION_MASK_FLAG);
        if (jl_atomic_load_relaxed(&bpart->min_world) > jl_require_world)
            goto invalidated;
    }
    {
        if (!jl_bkind_is_some_explicit_import(kind) && kind != PARTITION_KIND_IMPLICIT_GLOBAL)
            return 1;
        jl_binding_t *imported_binding = (jl_binding_t*)bpart->restriction;
        jl_binding_partition_t *latest_imported_bpart = jl_atomic_load_relaxed(&imported_binding->partitions);
        if (no_replacement)
            goto add_backedge;
        if (!latest_imported_bpart)
            return 1;
        if (jl_atomic_load_relaxed(&latest_imported_bpart->min_world) <=
            jl_atomic_load_relaxed(&bpart->min_world)) {
    add_backedge:
            // Imported binding is still valid
            if ((kind == PARTITION_KIND_EXPLICIT || kind == PARTITION_KIND_IMPORTED) &&
                    external_blob_index((jl_value_t*)imported_binding) != mod_idx) {
                jl_add_binding_backedge(imported_binding, (jl_value_t*)b);
            }
            return 1;
        }
        else {
            // Binding partition was invalidated
            assert(jl_atomic_load_relaxed(&bpart->min_world) == jl_require_world);
            jl_atomic_store_relaxed(&bpart->min_world,
                jl_atomic_load_relaxed(&latest_imported_bpart->min_world));
        }
    }
invalidated:
    // We need to go through and re-validate any bindings in the same image that
    // may have imported us.
    if (b->backedges) {
        JL_LOCK(&b->globalref->mod->lock);
        for (size_t i = 0; i < jl_array_len(b->backedges); i++) {
            jl_value_t *edge = jl_array_ptr_ref(b->backedges, i);
            if (!jl_is_binding(edge))
                continue;
            jl_binding_t *bedge = (jl_binding_t*)edge;
            if (!jl_atomic_load_relaxed(&bedge->partitions))
                continue;
            JL_UNLOCK(&b->globalref->mod->lock);
            jl_validate_binding_partition(bedge, jl_atomic_load_relaxed(&bedge->partitions), mod_idx, 0, 0);
            JL_LOCK(&b->globalref->mod->lock);
        }
        JL_UNLOCK(&b->globalref->mod->lock);
    }
    if (bpart->kind & PARTITION_FLAG_EXPORTED) {
        jl_module_t *mod = b->globalref->mod;
        jl_sym_t *name = b->globalref->name;
        JL_LOCK(&mod->lock);
        jl_atomic_store_release(&mod->export_set_changed_since_require_world, 1);
        if (mod->usings_backedges != jl_nothing) {
            for (size_t i = 0; i < jl_array_len(mod->usings_backedges); i++) {
                jl_module_t *edge = (jl_module_t*)jl_array_ptr_ref(mod->usings_backedges, i);
                jl_binding_t *importee = jl_get_module_binding(edge, name, 0);
                if (!importee)
                    continue;
                if (!jl_atomic_load_relaxed(&importee->partitions))
                    continue;
                JL_UNLOCK(&mod->lock);
                jl_validate_binding_partition(importee, jl_atomic_load_relaxed(&importee->partitions), mod_idx, 0, 0);
                JL_LOCK(&mod->lock);
            }
        }
        JL_UNLOCK(&mod->lock);
        return 0;
    }
    return 1;
}

static int all_usings_unchanged_implicit(jl_module_t *mod)
{
    int unchanged_implicit = 1;
    for (size_t i = 0; unchanged_implicit && i < module_usings_length(mod); i++) {
        jl_module_t *usee = module_usings_getmod(mod, i);
        unchanged_implicit &= !jl_atomic_load_acquire(&usee->export_set_changed_since_require_world);
    }
    return unchanged_implicit;
}

// Returns 0 on success, -1 when the restore was deliberately abandoned after the
// relink probe (JULIA_PKGIMAGE_RELINK with a rebuilt dependency): the abandonment
// happens before any relocation is applied or any global state is repointed, so the
// caller can simply refuse the cache and fall back to recompiling.
static int jl_restore_system_image_from_stream_(ios_t *f, jl_image_t *image,
                                                jl_array_t *depmods, uint64_t checksum,
                                /* outputs */   jl_array_t **restored,         jl_array_t **init_order,
                                                jl_array_t **extext_methods, jl_array_t **internal_methods,
                                                jl_array_t **new_ext_cis, jl_array_t **method_roots_list,
                                                pkgcachesizes *cachesizes) JL_GC_DISABLED
{
    jl_task_t *ct = jl_current_task;
    int en = jl_gc_enable(0);
    ios_t sysimg, const_data, symbols, relocs, gvar_record, fptr_record;
    jl_serializer_state s = {0};
    s.incremental = restored != NULL; // jl_linkage_blobs.len > 0;
    s.image = image;
    s.s = NULL;
    s.const_data = &const_data;
    s.symbols = &symbols;
    s.relocs = &relocs;
    s.gvar_record = &gvar_record;
    s.fptr_record = &fptr_record;
    s.ptls = ct->ptls;
    jl_value_t **_tags[NUM_TAGS];
    jl_value_t ***tags = s.incremental ? NULL : _tags;
    if (!s.incremental)
        get_tags(_tags);

    htable_t new_dt_objs;
    htable_new(&new_dt_objs, 0);
    arraylist_new(&deser_sym, 0);

    if (jl_options.use_sysimage_native_code != JL_OPTIONS_USE_SYSIMAGE_NATIVE_CODE_YES || IMAGE_NATIVE_CODE_TAINTED) {
        memset(&image->fptrs, 0, sizeof(image->fptrs));
        image->gvars_base = NULL;
        IMAGE_NATIVE_CODE_TAINTED = 1;
    }

    // step 1: read section map
    assert(ios_pos(f) == 0 && f->bm == bm_mem);
    size_t sizeof_sysdata = read_uint(f);
    ios_static_buffer(&sysimg, f->buf, sizeof_sysdata + sizeof(uintptr_t));
    ios_skip(f, sizeof_sysdata);

    size_t sizeof_constdata = read_uint(f);
    // realign stream to max-alignment for data
    ios_seek(f, LLT_ALIGN(ios_pos(f), JL_CACHE_BYTE_ALIGNMENT));
    ios_static_buffer(&const_data, f->buf + f->bpos, sizeof_constdata);
    ios_skip(f, sizeof_constdata);

    size_t sizeof_sysimg = f->bpos;

    size_t sizeof_symbols = read_uint(f);
    ios_seek(f, LLT_ALIGN(ios_pos(f), 8));
    ios_static_buffer(&symbols, f->buf + f->bpos, sizeof_symbols);
    ios_skip(f, sizeof_symbols);

    size_t sizeof_relocations = read_uint(f);
    ios_seek(f, LLT_ALIGN(ios_pos(f), 8));
    assert(!ios_eof(f));
    ios_static_buffer(&relocs, f->buf + f->bpos, sizeof_relocations);
    ios_skip(f, sizeof_relocations);

    size_t sizeof_gvar_record = read_uint(f);
    ios_seek(f, LLT_ALIGN(ios_pos(f), 8));
    assert(!ios_eof(f));
    ios_static_buffer(&gvar_record, f->buf + f->bpos, sizeof_gvar_record);
    ios_skip(f, sizeof_gvar_record);

    size_t sizeof_fptr_record = read_uint(f);
    ios_seek(f, LLT_ALIGN(ios_pos(f), 8));
    assert(!ios_eof(f));
    ios_static_buffer(&fptr_record, f->buf + f->bpos, sizeof_fptr_record);
    ios_skip(f, sizeof_fptr_record);

    // step 2: get references to special values
    ios_seek(f, LLT_ALIGN(ios_pos(f), 8));
    assert(!ios_eof(f));
    s.s = f;
    uintptr_t offset_restored = 0, offset_init_order = 0, offset_extext_methods = 0, offset_new_ext_cis = 0, offset_method_roots_list = 0;
    if (!s.incremental) {
        size_t i;
        for (i = 0; tags[i] != NULL; i++) {
            jl_value_t **tag = tags[i];
            *tag = jl_read_value(&s);
        }
        for (i = 0; i < jl_n_builtins; i++)
            jl_builtin_instances[i] = jl_read_value(&s);
#define XX(name, type) jl_##name = (type)jl_read_value(&s);
        JL_EXPORTED_DATA_POINTERS(XX)
#undef XX
#define XX(name, type) jl_##name = (type)jl_read_value(&s);
        JL_CONST_GLOBAL_VARS(XX)
#undef XX
#define XX(name) \
        ijl_small_typeof[(jl_##name##_tag << 4) / sizeof(*ijl_small_typeof)] = jl_##name##_type;
        JL_SMALL_TYPEOF(XX)
#undef XX
        export_jl_small_typeof();
        export_jl_sysimg_globals();
        jl_global_roots_list = (jl_genericmemory_t*)jl_read_value(&s);
        jl_global_roots_keyset = (jl_genericmemory_t*)jl_read_value(&s);
        s.ptls->root_task->tls = jl_read_value(&s);
        jl_gc_wb(s.ptls->root_task, s.ptls->root_task->tls);

        uint32_t gs_ctr = read_uint32(f);
        jl_require_world = read_uint(f);
        jl_atomic_store_release(&jl_world_counter, jl_require_world);
        jl_typeinf_world = read_uint(f);
        jl_set_gs_ctr(gs_ctr);
    }
    else {
        offset_restored = jl_read_offset(&s);
        offset_init_order = jl_read_offset(&s);
        offset_extext_methods = jl_read_offset(&s);
        offset_new_ext_cis = jl_read_offset(&s);
        offset_method_roots_list = jl_read_offset(&s);
    }
    s.buildid_depmods_idxs = depmod_to_imageidx(depmods);
    size_t nlinks_gctags = read_uint32(f);
    if (nlinks_gctags > 0) {
        s.link_ids_gctags = jl_alloc_array_1d(jl_array_int32_type, nlinks_gctags);
        ios_read(f, (char*)jl_array_data(s.link_ids_gctags, uint32_t), nlinks_gctags * sizeof(uint32_t));
    }
    size_t nlinks_relocs = read_uint32(f);
    if (nlinks_relocs > 0) {
        s.link_ids_relocs = jl_alloc_array_1d(jl_array_int32_type, nlinks_relocs);
        ios_read(f, (char*)jl_array_data(s.link_ids_relocs, uint32_t), nlinks_relocs * sizeof(uint32_t));
    }
    size_t nlinks_gvars = read_uint32(f);
    if (nlinks_gvars > 0) {
        s.link_ids_gvars = jl_alloc_array_1d(jl_array_int32_type, nlinks_gvars);
        ios_read(f, (char*)jl_array_data(s.link_ids_gvars, uint32_t), nlinks_gvars * sizeof(uint32_t));
    }
    size_t nlinks_external_fnvars = read_uint32(f);
    if (nlinks_external_fnvars > 0) {
        s.link_ids_external_fnvars = jl_alloc_array_1d(jl_array_int32_type, nlinks_external_fnvars);
        ios_read(f, (char*)jl_array_data(s.link_ids_external_fnvars, uint32_t), nlinks_external_fnvars * sizeof(uint32_t));
    }
    uint32_t external_fns_begin = read_uint32(f);
    jl_import_table_t *relink_tbl = NULL;
    int relink_ok = 0;
    {   // import table, written by jl_write_import_table
        size_t nimports = read_uint32(f);
        size_t nkeyed = 0, keybytes = 0;
        jl_import_table_t *itbl = NULL;
        int want_relink = getenv("JULIA_PKGIMAGE_RELINK") != NULL;
        if (want_relink && nimports) {
            itbl = (jl_import_table_t*)calloc(1, sizeof(jl_import_table_t));
            itbl->n = nimports;
            itbl->e = (jl_import_entry_t*)calloc(nimports, sizeof(jl_import_entry_t));
        }
        for (size_t i = 0; i < nimports; i++) {
            uint32_t depsidx = read_uint32(f);
            uint64_t digest = read_uint64(f);
            if (digest)
                nkeyed++;
            uint64_t off = read_uint64(f);
            size_t len = read_uint32(f);   // the locator, for re-deriving the object
            if (itbl) {
                itbl->e[i].depsidx = depsidx;
                itbl->e[i].digest = digest;
                itbl->e[i].offset = off;
                itbl->e[i].loclen = (uint32_t)len;
                if (len) {
                    itbl->e[i].loc = (char*)malloc_s(len + 1);
                    ios_read(f, itbl->e[i].loc, len);
                    itbl->e[i].loc[len] = '\0';
                    keybytes += len;
                    continue;
                }
            }
            if (len) {
                keybytes += len;
                ios_skip(f, len);
            }
        }
        if (itbl) {
            // Index the table by the (deps-index, offset) pair every reference carries.
            // Built before resolution because the `external_fns` gvars are named by the
            // same pair, and what those entries may resolve to is narrower.
            jl_relink_ent_t *map = (jl_relink_ent_t*)malloc_s(nimports * sizeof(jl_relink_ent_t));
            for (size_t i = 0; i < nimports; i++) {
                map[i].depsidx = itbl->e[i].depsidx;
                map[i].idx = (uint32_t)i;
                map[i].offset = itbl->e[i].offset;
                map[i].obj = NULL;
            }
            qsort(map, nimports, sizeof(jl_relink_ent_t), relink_ent_cmp);
            s.relink_map = map;
            s.relink_nmap = nimports;
            uint8_t *extfn = NULL;
            if (image->gvars_base != NULL) {
                // Which entries a gvar slot past `external_fns_begin` names -- decoded the
                // way `jl_update_all_gvars` will decode them, including the side table, so
                // the link index stays in step.
                extfn = (uint8_t*)calloc(nimports, 1);
                reloc_t *gvars = (reloc_t*)gvar_record.buf;
                size_t ngvars = gvar_record.size / sizeof(reloc_t);
                int li = 0;
                for (size_t i = external_fns_begin; i < ngvars; i++) {
                    uintptr_t rid = gvars[i];
                    enum RefTags tag = (enum RefTags)(rid >> RELOC_TAG_OFFSET);
                    size_t off = rid & (((uintptr_t)1 << RELOC_TAG_OFFSET) - 1);
                    size_t d;
                    if (tag == SysimageLinkage) {
#ifdef _P64
                        d = off >> DEPS_IDX_OFFSET;
                        off &= ((size_t)1 << DEPS_IDX_OFFSET) - 1;
#else
                        d = 0;
#endif
                    }
                    else if (tag == ExternalLinkage) {
                        assert(s.link_ids_external_fnvars && li < jl_array_len(s.link_ids_external_fnvars));
                        d = jl_array_data(s.link_ids_external_fnvars, uint32_t)[li++];
                    }
                    else {
                        continue;   // an entry point of this image's own; not an import
                    }
                    jl_relink_ent_t *ent = relink_lookup(&s, d, off);
                    if (ent != NULL)
                        extfn[ent->idx] = 1;
                }
            }
            relink_ok = jl_relink_probe(&s, itbl, depmods, extfn);
            free(extfn);
            if (relink_ok && relink_probe_buildid_mismatch) {
                for (size_t i = 0; i < nimports; i++)
                    map[i].obj = itbl->e[map[i].idx].resolved;
                // arming this is what turns an accepted entry into an avoided rebuild
                s.relink_deps = relink_mismatched_deps;
                s.relink_ndeps = relink_mismatched_ndeps;
            }
            relink_tbl = itbl;
        }
        if (getenv("JULIA_IMPORT_KEYS"))
            jl_safe_printf("IMPORTKEYS_READ entries=%zu keyed=%zu keybytes=%zu\n",
                           nimports, nkeyed, keybytes);
    }
    if (s.incremental && relink_probe_buildid_mismatch && !relink_ok) {
        // A dependency was rebuilt with a different build_id, and at least one object this
        // image imports from it could not be found again. The relocations below would be
        // applied against the wrong blob layout, so abandon the restore before touching any
        // global state -- resolution is a pre-pass precisely so this is still possible.
        // The caller refuses the cache and Julia recompiles.
        relink_free(&s, relink_tbl);
        ios_close(&sysimg);
        ios_close(&const_data);
        ios_close(&symbols);
        ios_close(&relocs);
        ios_close(&gvar_record);
        ios_close(&fptr_record);
        htable_free(&new_dt_objs);
        arraylist_free(&deser_sym);
        jl_gc_enable(en);
        return -1;
    }
    if (s.incremental) {
        assert(restored && init_order && extext_methods && internal_methods && new_ext_cis && method_roots_list);
        *restored = (jl_array_t*)jl_delayed_reloc(&s, offset_restored);
        *init_order = (jl_array_t*)jl_delayed_reloc(&s, offset_init_order);
        *extext_methods = (jl_array_t*)jl_delayed_reloc(&s, offset_extext_methods);
        *new_ext_cis = (jl_array_t*)jl_delayed_reloc(&s, offset_new_ext_cis);
        *method_roots_list = (jl_array_t*)jl_delayed_reloc(&s, offset_method_roots_list);
        *internal_methods = jl_alloc_vec_any(0);
    }
    s.s = NULL;

    // step 3: apply relocations
    assert(!ios_eof(f));
    jl_read_symbols(&s);
    ios_close(&symbols);

    char *image_base = (char*)&sysimg.buf[0];
    reloc_t *relocs_base = (reloc_t*)&relocs.buf[0];

    s.s = &sysimg;
    jl_read_reloclist(&s, s.link_ids_gctags, GC_OLD | GC_IN_IMAGE); // gctags
    size_t sizeof_tags = ios_pos(&relocs);
    (void)sizeof_tags;
    jl_read_reloclist(&s, s.link_ids_relocs, 0); // general relocs
    jl_read_memreflist(&s); // memowner_list relocs (must come before memref_list reads the pointers and after general relocs computes the pointers)
    jl_read_memreflist(&s); // memref_list relocs
    // s.link_ids_gvars will be processed in `jl_update_all_gvars`
    // s.link_ids_external_fns will be processed in `jl_update_all_gvars`
    jl_update_all_gvars(&s, image, external_fns_begin); // gvars relocs
    if (s.incremental) {
        jl_read_arraylist(s.relocs, &s.uniquing_types);
        jl_read_arraylist(s.relocs, &s.uniquing_objs);
        jl_read_arraylist(s.relocs, &s.fixup_types);
    }
    else {
        arraylist_new(&s.uniquing_types, 0);
        arraylist_new(&s.uniquing_objs, 0);
        arraylist_new(&s.fixup_types, 0);
    }
    jl_read_arraylist(s.relocs, &s.fixup_objs);
    // Perform the uniquing of objects that we don't "own" and consequently can't promise
    // weren't created by some other package before this one got loaded:
    // - iterate through all objects that need to be uniqued. The first encounter has to be the
    //   "reconstructable blob". We either look up the object (if something has created it previously)
    //   or construct it for the first time, crucially outside the pointer range of any pkgimage.
    //   This ensures it stays unique-worthy.
    // - after we've stored the address of the "real" object (which for convenience we do among the data
    //   written to allow lookup/reconstruction), then we have to update references to that "reconstructable blob":
    //   instead of performing the relocation within the package image, we instead (re)direct all references
    //   to the external object.
    arraylist_t cleanup_list;
    arraylist_new(&cleanup_list, 0);
    arraylist_t delay_list;
    arraylist_new(&delay_list, 0);
    JL_LOCK(&typecache_lock); // Might GC--prevent other threads from changing any type caches while we inspect them all
    for (size_t i = 0; i < s.uniquing_types.len; i++) {
        uintptr_t item = (uintptr_t)s.uniquing_types.items[i];
        // check whether we are operating on the typetag
        // (needing to ignore GC bits) or a regular field
        // and check whether this is a gvar index
        int tag = (item & 3);
        item &= ~(uintptr_t)3;
        uintptr_t *pfld;
        jl_value_t **obj, *newobj;
        if (tag == 3) {
            obj = (jl_value_t**)(image_base + item);
            pfld = NULL;
            for (size_t i = 0; i < delay_list.len; i += 2) {
                if (obj == (jl_value_t **)delay_list.items[i + 0]) {
                    pfld = (uintptr_t*)delay_list.items[i + 1];
                    delay_list.items[i + 1] = arraylist_pop(&delay_list);
                    delay_list.items[i + 0] = arraylist_pop(&delay_list);
                    break;
                }
            }
            assert(pfld);
        }
        else if (tag == 2) {
            if (image->gvars_base == NULL)
                continue;
            item >>= 2;
            assert(item < s.gvar_record->size / sizeof(reloc_t));
            pfld = sysimg_gvars(image->gvars_base, image->gvars_offsets, item);
            obj = *(jl_value_t***)pfld;
        }
        else {
            pfld = (uintptr_t*)(image_base + item);
            if (tag == 1)
                obj = (jl_value_t**)jl_typeof(jl_valueof(pfld));
            else
                obj = *(jl_value_t***)pfld;
            if ((char*)obj > (char*)pfld) {
                // this must be the super field
                assert(tag == 0);
                arraylist_push(&delay_list, obj);
                arraylist_push(&delay_list, pfld);
                ptrhash_put(&new_dt_objs, (void*)obj, obj); // mark obj as invalid
                *pfld = (uintptr_t)NULL;
                continue;
            }
        }
        uintptr_t otyp = jl_typetagof(obj);   // the original type of the object that was written here
        assert(image_base < (char*)obj && (char*)obj <= image_base + sizeof_sysimg);
        if (otyp == jl_datatype_tag << 4) {
            jl_datatype_t *dt = (jl_datatype_t*)obj[0], *newdt;
            if (jl_is_datatype(dt)) {
                newdt = dt; // already done
            }
            else {
                dt = (jl_datatype_t*)obj;
                arraylist_push(&cleanup_list, (void*)obj);
                ptrhash_remove(&new_dt_objs, (void*)obj); // unmark obj as invalid before must_be_new_dt
                if (must_be_new_dt((jl_value_t*)dt, &new_dt_objs, image_base, sizeof_sysimg))
                    newdt = NULL;
                else
                    newdt = jl_lookup_cache_type_(dt);
                if (newdt == NULL) {
                    // make a non-owned copy of obj so we don't accidentally
                    // assume this is the unique copy later
                    newdt = jl_new_uninitialized_datatype();
                    jl_astaggedvalue(newdt)->bits.gc = GC_OLD;
                    // leave most fields undefined for now, but we may need instance later,
                    // and we overwrite the name field (field 0) now so preserve it too
                    if (dt->instance) {
                        if (dt->instance == jl_nothing)
                            dt->instance = jl_gc_permobj(ct->ptls, 0, newdt, 0);
                        newdt->instance = dt->instance;
                    }
                    static_assert(offsetof(jl_datatype_t, name) == 0, "");
                    newdt->name = dt->name;
                    ptrhash_put(&new_dt_objs, (void*)newdt, dt);
                }
                else {
                    assert(newdt->hash == dt->hash);
                }
                obj[0] = (jl_value_t*)newdt;
            }
            newobj = (jl_value_t*)newdt;
        }
        else {
            assert(!(image_base < (char*)otyp && (char*)otyp <= image_base + sizeof_sysimg));
            newobj = ((jl_datatype_t*)otyp)->instance;
            assert(newobj && newobj != jl_nothing);
            arraylist_push(&cleanup_list, (void*)obj);
        }
        if (tag == 1)
            *pfld = (uintptr_t)newobj | GC_OLD | GC_IN_IMAGE;
        else
            *pfld = (uintptr_t)newobj;
        assert(!(image_base < (char*)newobj && (char*)newobj <= image_base + sizeof_sysimg));
        assert(jl_typetagis(obj, otyp));
    }
    assert(delay_list.len == 0);
    arraylist_free(&delay_list);
    // now that all the fields of dt are assigned and unique, copy them into
    // their final newdt memory location: this ensures we do not accidentally
    // think this pkg image has the singular unique copy of it
    void **table = new_dt_objs.table;
    for (size_t i = 0; i < new_dt_objs.size; i += 2) {
        void *dt = table[i + 1];
        if (dt != HT_NOTFOUND) {
            jl_datatype_t *newdt = (jl_datatype_t*)table[i];
            jl_typename_t *name = newdt->name;
            static_assert(offsetof(jl_datatype_t, name) == 0, "");
            assert(*(void**)dt == (void*)newdt);
            *newdt = *(jl_datatype_t*)dt; // copy the datatype fields (except field 1, which we corrupt above)
            newdt->name = name;
        }
    }
    // we should never see these pointers again, so scramble their memory, so any attempt to look at them crashes
    for (size_t i = 0; i < cleanup_list.len; i++) {
        void *item = cleanup_list.items[i];
        jl_taggedvalue_t *o = jl_astaggedvalue(item);
        jl_value_t *t = jl_typeof(item); // n.b. might be 0xbabababa already
        if (t == (jl_value_t*)jl_datatype_type)
            memset(o, 0xba, sizeof(jl_value_t*) + sizeof(jl_datatype_t));
        else
            memset(o, 0xba, sizeof(jl_value_t*) + 0); // singleton
        o->bits.in_image = 1;
    }
    arraylist_grow(&cleanup_list, -cleanup_list.len);
    // finally cache all our new types now
    jl_safepoint_suspend_all_threads(ct); // past this point, it is now not safe to observe the intermediate states on other threads via reflection, so temporarily pause those
    for (size_t i = 0; i < new_dt_objs.size; i += 2) {
        void *dt = table[i + 1];
        if (dt != HT_NOTFOUND) {
            jl_datatype_t *newdt = (jl_datatype_t*)table[i];
            jl_cache_type_(newdt);
        }
    }
    for (size_t i = 0; i < s.fixup_types.len; i++) {
        uintptr_t item = (uintptr_t)s.fixup_types.items[i];
        jl_value_t *obj = (jl_value_t*)(image_base + item);
        assert(jl_is_datatype(obj));
        jl_cache_type_((jl_datatype_t*)obj);
    }
    JL_UNLOCK(&typecache_lock); // Might GC
    jl_safepoint_resume_all_threads(ct); // TODO: move this later to also protect MethodInstance allocations, but we would need to acquire all jl_specializations_get_linfo and jl_module_globalref locks, which is hard
    // Perform fixups: things like updating world ages, inserting methods & specializations, etc.
    for (size_t i = 0; i < s.uniquing_objs.len; i++) {
        uintptr_t item = (uintptr_t)s.uniquing_objs.items[i];
        // check whether this is a gvar index
        int tag = (item & 3);
        assert(tag == 0 || tag == 2);
        item &= ~(uintptr_t)3;
        uintptr_t *pfld;
        jl_value_t **obj, *newobj;
        if (tag == 2) {
            if (image->gvars_base == NULL)
                continue;
            item >>= 2;
            assert(item < s.gvar_record->size / sizeof(reloc_t));
            pfld = sysimg_gvars(image->gvars_base, image->gvars_offsets, item);
            obj = *(jl_value_t***)pfld;
        }
        else {
            pfld = (uintptr_t*)(image_base + item);
            obj = *(jl_value_t***)pfld;
        }
        uintptr_t otyp = jl_typetagof(obj);   // the original type of the object that was written here
        if (otyp == (uintptr_t)jl_method_instance_type) {
            assert(image_base < (char*)obj && (char*)obj <= image_base + sizeof_sysimg);
            jl_value_t *m = obj[0];
            if (jl_is_method_instance(m)) {
                newobj = m; // already done
            }
            else {
                arraylist_push(&cleanup_list, (void*)obj);
                jl_value_t *specTypes = obj[1];
                jl_value_t *sparams = obj[2];
                newobj = (jl_value_t*)jl_specializations_get_linfo((jl_method_t*)m, specTypes, (jl_svec_t*)sparams);
                obj[0] = newobj;
            }
        }
        else if (otyp == (uintptr_t)jl_binding_type) {
            jl_value_t *m = obj[0];
            if (jl_is_binding(m)) {
                newobj = m; // already done
            }
            else {
                arraylist_push(&cleanup_list, (void*)obj);
                jl_value_t *name = obj[1];
                newobj = (jl_value_t*)jl_get_module_binding((jl_module_t*)m, (jl_sym_t*)name, 1);
                obj[0] = newobj;
            }
        }
        else {
            abort(); // should be unreachable
        }
        *pfld = (uintptr_t)newobj;
        assert(!(image_base < (char*)newobj && (char*)newobj <= image_base + sizeof_sysimg));
        assert(jl_typetagis(obj, otyp));
    }
    arraylist_free(&s.uniquing_types);
    arraylist_free(&s.uniquing_objs);
    for (size_t i = 0; i < cleanup_list.len; i++) {
        void *item = cleanup_list.items[i];
        jl_taggedvalue_t *o = jl_astaggedvalue(item);
        jl_value_t *t = jl_typeof(item);
        if (t == (jl_value_t*)jl_method_instance_type)
            memset(o, 0xba, sizeof(jl_value_t*) * 3); // only specTypes and sparams fields stored
        else if (t == (jl_value_t*)jl_binding_type)
            memset(o, 0xba, sizeof(jl_value_t*) * 3); // stored as mod/name
        o->bits.in_image = 1;
    }
    arraylist_free(&cleanup_list);
    for (size_t i = 0; i < s.fixup_objs.len; i++) {
        uintptr_t item = (uintptr_t)s.fixup_objs.items[i];
        jl_value_t *obj = (jl_value_t*)(image_base + item);
        if (jl_typetagis(obj, jl_typemap_entry_type) || jl_is_method(obj) || jl_is_code_instance(obj)) {
            jl_array_ptr_1d_push(*internal_methods, obj);
            assert(s.incremental);
        }
        else if (jl_is_method_instance(obj)) {
            jl_method_instance_t *newobj = jl_specializations_get_or_insert((jl_method_instance_t*)obj);
            assert(newobj == (jl_method_instance_t*)obj); // strict insertion expected
            (void)newobj;
        }
        else if (jl_is_globalref(obj)) {
            jl_globalref_t *r = (jl_globalref_t*)obj;
            if (r->binding == NULL) {
                jl_globalref_t *gr = (jl_globalref_t*)jl_module_globalref(r->mod, r->name);
                r->binding = gr->binding;
                jl_gc_wb(r, gr->binding);
            }
        }
        else if (jl_is_module(obj)) {
            // rebuild the usings table for module v
            // TODO: maybe want to hold the lock on `v`, but that only strongly matters for async / thread safety
            // and we are already bad at that
            jl_module_t *mod = (jl_module_t*)obj;
            mod->build_id.hi = checksum;
            if (mod->usings.items != &mod->usings._space[0]) {
                // arraylist_t assumes we called malloc to get this memory, so make that true now
                void **newitems = (void**)malloc_s(mod->usings.max * sizeof(void*));
                memcpy(newitems, mod->usings.items, mod->usings.len * sizeof(void*));
                mod->usings.items = newitems;
            }
            size_t mod_idx = external_blob_index((jl_value_t*)mod);
            if (s.incremental) {
                // Rebuild cross-image usings backedges
                for (size_t i = 0; i < module_usings_length(mod); ++i) {
                    struct _jl_module_using *data = module_usings_getidx(mod, i);
                    if (external_blob_index((jl_value_t*)data->mod) != mod_idx) {
                        jl_add_usings_backedge(data->mod, mod);
                    }
                }
            }
        }
        else {
            abort();
        }
    }
    if (s.incremental) {
        int no_replacement = jl_atomic_load_relaxed(&jl_first_image_replacement_world) == ~(size_t)0;
        for (size_t i = 0; i < s.fixup_objs.len; i++) {
            uintptr_t item = (uintptr_t)s.fixup_objs.items[i];
            jl_value_t *obj = (jl_value_t*)(image_base + item);
            if (jl_is_module(obj)) {
                jl_module_t *mod = (jl_module_t*)obj;
                size_t mod_idx = external_blob_index((jl_value_t*)mod);
                jl_svec_t *table = jl_atomic_load_relaxed(&mod->bindings);
                int unchanged_implicit = no_replacement || all_usings_unchanged_implicit(mod);
                for (size_t i = 0; i < jl_svec_len(table); i++) {
                    jl_binding_t *b = (jl_binding_t*)jl_svecref(table, i);
                    if ((jl_value_t*)b == jl_nothing)
                        continue;
                    jl_binding_partition_t *bpart = jl_atomic_load_relaxed(&b->partitions);
                    if (!jl_validate_binding_partition(b, bpart, mod_idx, unchanged_implicit, no_replacement)) {
                        unchanged_implicit = all_usings_unchanged_implicit(mod);
                    }
                }
            }
        }
    }
    arraylist_free(&s.fixup_types);
    arraylist_free(&s.fixup_objs);

    if (s.incremental)
        jl_root_new_gvars(&s, image, external_fns_begin);
    ios_close(&relocs);
    ios_close(&const_data);
    ios_close(&gvar_record);

    htable_free(&new_dt_objs);

    s.s = NULL;

    if (0) {
        printf("sysimg size breakdown:\n"
               "     sys data: %8u\n"
               "  isbits data: %8u\n"
               "      symbols: %8u\n"
               "    tags list: %8u\n"
               "   reloc list: %8u\n"
               "    gvar list: %8u\n"
               "    fptr list: %8u\n",
            (unsigned)sizeof_sysdata,
            (unsigned)sizeof_constdata,
            (unsigned)sizeof_symbols,
            (unsigned)sizeof_tags,
            (unsigned)(sizeof_relocations - sizeof_tags),
            (unsigned)sizeof_gvar_record,
            (unsigned)sizeof_fptr_record);
    }
    if (cachesizes) {
        cachesizes->sysdata = sizeof_sysdata;
        cachesizes->isbitsdata = sizeof_constdata;
        cachesizes->symboldata = sizeof_symbols;
        cachesizes->tagslist = sizeof_tags;
        cachesizes->reloclist = sizeof_relocations - sizeof_tags;
        cachesizes->gvarlist = sizeof_gvar_record;
        cachesizes->fptrlist = sizeof_fptr_record;
    }

    s.s = &sysimg;
    jl_update_all_fptrs(&s, image); // fptr relocs and registration
    s.s = NULL;

    ios_close(&fptr_record);
    ios_close(&sysimg);

    if (!s.incremental)
        jl_gc_reset_alloc_count();
    arraylist_free(&deser_sym);

    // Prepare for later external linkage against the sysimg
    // Also sets up images for protection against garbage collection
    arraylist_push(&jl_linkage_blobs, (void*)image_base);
    arraylist_push(&jl_linkage_blobs, (void*)(image_base + sizeof_sysimg));
    arraylist_push(&jl_image_relocs, (void*)relocs_base);
    if (restored == NULL) {
        arraylist_push(&jl_top_mods, (void*)jl_top_module);
    } else {
        size_t len = jl_array_nrows(*restored);
        assert(len > 0);
        jl_module_t *topmod = (jl_module_t*)jl_array_ptr_ref(*restored, len-1);
        // Ordinarily set during deserialization, but our compiler stub image,
        // just returns a reference to the sysimage version, so we set it here.
        topmod->build_id.hi = checksum;
        assert(jl_is_module(topmod));
        arraylist_push(&jl_top_mods, (void*)topmod);
    }
    jl_timing_counter_inc(JL_TIMING_COUNTER_ImageSize, sizeof_sysimg + sizeof(uintptr_t));
    rebuild_image_blob_tree();
    relink_free(&s, relink_tbl);   // every relocation that could consult it has run

    // jl_printf(JL_STDOUT, "%ld blobs to link against\n", jl_linkage_blobs.len >> 1);
    jl_gc_enable(en);

    if (s.incremental)
        jl_add_methods(*extext_methods);
    return 0;
}

static jl_value_t *jl_validate_cache_file(ios_t *f, jl_array_t *depmods, uint64_t *checksum, int64_t *dataendpos, int64_t *datastartpos)
{
    uint8_t pkgimage = 0;
    if (ios_eof(f) || 0 == (*checksum = jl_read_verify_header(f, &pkgimage, dataendpos, datastartpos)) || (*checksum >> 32 != 0xfafbfcfd)) {
        return jl_get_exceptionf(jl_errorexception_type,
                "Precompile file header verification checks failed.");
    }
    uint8_t flags = read_uint8(f);
    if (pkgimage && !jl_match_cache_flags_current(flags)) {
        return jl_get_exceptionf(jl_errorexception_type, "Pkgimage flags mismatch");
    }
    if (!pkgimage) {
        // skip past the worklist
        size_t len;
        while ((len = read_int32(f)))
            ios_skip(f, len + 3 * sizeof(uint64_t));
        // skip past the dependency list
        size_t deplen = read_uint64(f);
        ios_skip(f, deplen - sizeof(uint64_t));
        read_uint64(f); // where is this write coming from?
    }

    // verify that the system state is valid
    return read_verify_mod_list(f, depmods);
}

// TODO?: refactor to make it easier to create the "package inspector"
static jl_value_t *jl_restore_package_image_from_stream(ios_t *f, jl_image_t *image, jl_array_t *depmods, int completeinfo, const char *pkgname, int needs_permalloc)
{
    JL_TIMING(LOAD_IMAGE, LOAD_Pkgimg);
    jl_timing_printf(JL_TIMING_DEFAULT_BLOCK, pkgname);
    uint64_t checksum = 0;
    int64_t dataendpos = 0;
    int64_t datastartpos = 0;
    jl_value_t *verify_fail = jl_validate_cache_file(f, depmods, &checksum, &dataendpos, &datastartpos);

    if (verify_fail)
        return verify_fail;

    assert(datastartpos > 0 && datastartpos < dataendpos);
    needs_permalloc = jl_options.permalloc_pkgimg || needs_permalloc;

    jl_value_t *restored = NULL;
    jl_array_t *init_order = NULL, *extext_methods = NULL, *internal_methods = NULL, *new_ext_cis = NULL, *method_roots_list = NULL;
    jl_svec_t *cachesizes_sv = NULL;
    JL_GC_PUSH7(&restored, &init_order, &extext_methods, &internal_methods, &new_ext_cis, &method_roots_list, &cachesizes_sv);

    { // make a permanent in-memory copy of f (excluding the header)
        ios_bufmode(f, bm_none);
        JL_SIGATOMIC_BEGIN();
        size_t len = dataendpos - datastartpos;
        char *sysimg;
        int success = !needs_permalloc;
        ios_seek(f, datastartpos);
        if (needs_permalloc)
            sysimg = (char*)jl_gc_perm_alloc(len, 0, 64, 0);
        else
            sysimg = &f->buf[f->bpos];
        if (needs_permalloc)
            success = ios_readall(f, sysimg, len) == len;
        if (!success) {
            restored = jl_get_exceptionf(jl_errorexception_type, "Error reading package image file.");
            JL_SIGATOMIC_END();
        }
        else {
            if (needs_permalloc)
                ios_close(f);
            ios_static_buffer(f, sysimg, len);
            pkgcachesizes cachesizes;
            int abandoned = jl_restore_system_image_from_stream_(f, image, depmods, checksum, (jl_array_t**)&restored, &init_order, &extext_methods, &internal_methods, &new_ext_cis, &method_roots_list, &cachesizes);
            JL_SIGATOMIC_END();
            if (abandoned) {
                // relink probe ran against a rebuilt dependency; refuse the cache so
                // nothing stale executes -- the caller falls back to recompiling
                restored = jl_get_exceptionf(jl_errorexception_type,
                        "Refusing cache for %s after relink probe: dependency build_id mismatch.", pkgname);
                JL_GC_POP();
                return restored;
            }

            // Add roots to methods
            int failed = jl_copy_roots(method_roots_list, jl_worklist_key((jl_array_t*)restored));
            if (failed != 0) {
                jl_printf(JL_STDERR, "Error copying roots to methods from Module: %s\n", pkgname);
                abort();
            }
            // Insert method extensions and handle edges
            int new_methods = jl_array_nrows(extext_methods) > 0;
            if (!new_methods) {
                size_t i, l = jl_array_nrows(internal_methods);
                for (i = 0; i < l; i++) {
                    jl_value_t *obj = jl_array_ptr_ref(internal_methods, i);
                    if (jl_is_method(obj)) {
                        new_methods = 1;
                        break;
                    }
                }
            }
            JL_LOCK(&world_counter_lock);
            // allocate a world for the new methods, and insert them there, invalidating content as needed
            size_t world = jl_atomic_load_relaxed(&jl_world_counter);
            if (new_methods)
                world += 1;
            jl_activate_methods(extext_methods, internal_methods, world, pkgname);
            // TODO: inject internal_methods into caches here, so the system can see them immediately as potential candidates (before validation)
            // allow users to start running in this updated world
            if (new_methods)
                jl_atomic_store_release(&jl_world_counter, world);
            // now permit more methods to be added again
            JL_UNLOCK(&world_counter_lock);

            if (completeinfo) {
                cachesizes_sv = jl_alloc_svec(7);
                jl_svecset(cachesizes_sv, 0, jl_box_long(cachesizes.sysdata));
                jl_svecset(cachesizes_sv, 1, jl_box_long(cachesizes.isbitsdata));
                jl_svecset(cachesizes_sv, 2, jl_box_long(cachesizes.symboldata));
                jl_svecset(cachesizes_sv, 3, jl_box_long(cachesizes.tagslist));
                jl_svecset(cachesizes_sv, 4, jl_box_long(cachesizes.reloclist));
                jl_svecset(cachesizes_sv, 5, jl_box_long(cachesizes.gvarlist));
                jl_svecset(cachesizes_sv, 6, jl_box_long(cachesizes.fptrlist));
                restored = (jl_value_t*)jl_svec(5, restored, init_order, internal_methods, method_roots_list, cachesizes_sv);
            }
            else {
                restored = (jl_value_t*)jl_svec(3, restored, init_order, internal_methods);
            }
        }
    }

    JL_GC_POP();
    return restored;
}

static void jl_restore_system_image_from_stream(ios_t *f, jl_image_t *image, uint32_t checksum)
{
    JL_TIMING(LOAD_IMAGE, LOAD_Sysimg);
    jl_restore_system_image_from_stream_(f, image, NULL, checksum | ((uint64_t)0xfdfcfbfa << 32), NULL, NULL, NULL, NULL, NULL, NULL, NULL);
}

JL_DLLEXPORT jl_value_t *jl_restore_incremental_from_buf(jl_image_buf_t buf, jl_image_t *image, jl_array_t *depmods, int completeinfo, const char *pkgname, int needs_permalloc)
{
    ios_t f;
    ios_static_buffer(&f, (char*)buf.data, buf.size);
    jl_value_t *ret = jl_restore_package_image_from_stream(&f, image, depmods, completeinfo, pkgname, needs_permalloc);
    ios_close(&f);
    return ret;
}

JL_DLLEXPORT jl_value_t *jl_restore_incremental(const char *fname, jl_array_t *depmods, int completeinfo, const char *pkgname)
{
    ios_t f;
    if (ios_file(&f, fname, 1, 0, 0, 0) == NULL) {
        return jl_get_exceptionf(jl_errorexception_type,
            "Cache file \"%s\" not found.\n", fname);
    }
    jl_image_t pkgimage = {};
    jl_value_t *ret = jl_restore_package_image_from_stream(&f, &pkgimage, depmods, completeinfo, pkgname, 1);
    ios_close(&f);
    return ret;
}

JL_DLLEXPORT void jl_restore_system_image(jl_image_t *image, jl_image_buf_t buf)
{
    ios_t f;

    if (buf.kind == JL_IMAGE_KIND_NONE)
        return;

    if (buf.kind == JL_IMAGE_KIND_SO)
        assert(image->fptrs.ptrs); // jl_init_processor_sysimg should already be run

    JL_SIGATOMIC_BEGIN();
    ios_static_buffer(&f, (char *)buf.data, buf.size);

    jl_restore_system_image_from_stream(&f, image, buf.checksum);

    ios_close(&f);
    JL_SIGATOMIC_END();
}

JL_DLLEXPORT jl_value_t *jl_restore_package_image_from_file(const char *fname, jl_array_t *depmods, int completeinfo, const char *pkgname, int ignore_native)
{
    void *pkgimg_handle = jl_dlopen(fname, JL_RTLD_LAZY);
    if (!pkgimg_handle) {
#ifdef _OS_WINDOWS_
        int err;
        char reason[256];
        err = GetLastError();
        win32_formatmessage(err, reason, sizeof(reason));
#else
        const char *reason = dlerror();
#endif
        jl_errorf("Error opening package file %s: %s\n", fname, reason);
    }

    jl_image_buf_t buf = get_image_buf(pkgimg_handle, /* is_pkgimage */ 1);

    jl_gc_notify_image_load(buf.data, buf.size);

    // Despite the name, this function actually parses the pkgimage
    jl_image_t pkgimage = jl_init_processor_pkgimg(buf);

    if (ignore_native) {
        // Must disable using native code in possible downstream users of this code:
        // https://github.com/JuliaLang/julia/pull/52123#issuecomment-1959965395.
        // The easiest way to do that is to disable it in all of them.
        IMAGE_NATIVE_CODE_TAINTED = 1;
    }

    jl_value_t* mod = jl_restore_incremental_from_buf(buf, &pkgimage, depmods, completeinfo, pkgname, 0);

    return mod;
}


#ifdef __cplusplus
}
#endif
