#include <stdlib.h>
#include <stdio.h>
#include "../rle.h"

int main()
{
    /* Iteration */
    rle_iter_state state = rle_iter_init(22);
    int i = 0;
    while (rle_iter_increment(&state, 5, NULL, 0)) {
        assert(state.key == 22);
        assert(state.i == i);
        i++;
    }

    uint64_t rletable1[4] = {-1, 2, 22, 3};
    state = rle_iter_init(22);
    i = 0;
    while (rle_iter_increment(&state, 5, rletable1, 4)) {
        assert(state.key == (i < 2 ? 22 : (i < 3 ? -1 : 22)));
        assert(state.i == i);
        i++;
    }

    uint64_t rletable2[4] = {-1, 0, 22, 3};
    state = rle_iter_init(22);
    i = 0;
    while (rle_iter_increment(&state, 5, rletable2, 4)) {
        assert(state.key == (i < 3 ? -1 : 22));
        assert(state.i == i);
        i++;
    }

    state = rle_iter_init(22);
    i = 0;
    while (rle_iter_increment(&state, 0, rletable2, 4)) {
        abort();
    }

    /* Indexing */
    rle_reference rr;
    uint64_t rletable3[8] = {0, 0, 5, 2, 22, 3, 0, 5};
    uint64_t keys3[7] = {0, 0, 5, 22, 22, 0, 0};
    int counts3[7] = {0, 1, 0, 0, 1, 2, 3};
    for (i = 0; i < 7; i++) {
        rle_index_to_reference(&rr, i, rletable3, 8, 0);
        assert(rr.key == keys3[i]);
        assert(rr.index == counts3[i]);
        assert(rle_reference_to_index(&rr, rletable3, 8, 0) == i);
    }
    uint64_t rletable4[6] = {5, 2, 22, 3, 0, 5};  // implicit first block
    for (i = 0; i < 7; i++) {
        rle_index_to_reference(&rr, i, rletable4, 6, 0);
        assert(rr.key == keys3[i]);
        assert(rr.index == counts3[i]);
        assert(rle_reference_to_index(&rr, rletable4, 6, 0) == i);
    }

    /* a reference the table cannot satisfy is reported, not answered with a stray index */
    rr.key = 99; rr.index = 0;   // no run has this key
    assert(rle_reference_to_index(&rr, rletable3, 8, 0) == RLE_NOTFOUND);
    rr.key = 5; rr.index = 1;    // key 5 has one item, not two
    assert(rle_reference_to_index(&rr, rletable3, 8, 0) == RLE_NOTFOUND);
    rr.key = 22; rr.index = 2;   // key 22 has two items, not three
    assert(rle_reference_to_index(&rr, rletable3, 8, 0) == RLE_NOTFOUND);
    rr.key = 1; rr.index = 0;    // no table at all, so only key0 is answerable
    assert(rle_reference_to_index(&rr, NULL, 0, 0) == RLE_NOTFOUND);

    return 0;
}
