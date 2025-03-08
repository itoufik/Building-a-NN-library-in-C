#define NN_IMPLEMENTATION
#include "nn.h"
#include <time.h>
#define BITS 8
#define rate 1e-1
#define train_epoch 70 * 1000

int main(void) {

  // Random seed
  srand(time(0));

  size_t n = (1 << BITS);
  size_t rows = n * n;
  Mat ti = mat_alloc(rows, 2 * BITS);
  Mat to = mat_alloc(rows, BITS + 1);
  for (size_t i = 0; i < ti.rows; i++) {
    size_t x = i / n;
    size_t y = i % n;
    size_t z = x + y;
    size_t overflow = z >= n;
    for (size_t j = 0; j < BITS; j++) {
      MAT_AT(ti, i, j) = (x >> j) & 1;
      MAT_AT(ti, i, j + BITS) = (y >> j) & 1;
      if (overflow) {
        MAT_AT(to, i, j) = 0;
      } else {
        MAT_AT(to, i, j) = (z >> j) & 1;
      }
    }
    MAT_AT(to, i, BITS) = overflow;
  }
  // MAT_PRINT(ti);
  // MAT_PRINT(to);

  size_t arch[] = {2 * BITS, 2 * BITS + 1,
                   BITS + 1}; // BITS number of inner layer
  NN nn = nn_alloc(arch, ARRAY_LEN(arch));
  NN g = nn_alloc(arch, ARRAY_LEN(arch));
  nn_rand(nn, 1, 0);
  // NN_PRINT(nn);
  FILE *fp = fopen("loss_data.csv", "w");
  if (!fp) {
    perror("Failed to open file");
    exit(EXIT_FAILURE);
  }
  // Write CSV header
  fprintf(fp, "iteration,cost\n");
  printf("cost before = %f\n", nn_cost(nn, ti, to));
  clock_t start = clock(); // start timer
  for (size_t i = 0; i < train_epoch; i++) {
    nn_backprop(nn, g, ti, to);
    nn_learn(nn, g, rate);
    if ((i % 500 == 0) || (i == train_epoch - 1)) {
      float cost = nn_cost(nn, ti, to);
      printf("cost after %zu steps = %f\n", i, cost);
      fprintf(fp, "%zu,%f\n", i, cost);
    }
  }
  fclose(fp);
  clock_t end = clock(); // end timer
  double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Execution time  for training loop: %f seconds\n", time_taken);

  for (size_t x = 0; x < n; x++) {
    for (size_t y = 0; y < n; y++) // FIXED: Iterate `y` correctly up to `n - 1`
    {
      printf("%zu + %zu = ", x, y);
      for (size_t j = 0; j < BITS; j++) {
        MAT_AT(NN_INPUT(nn), 0, j) = (x >> j) & 1;
        MAT_AT(NN_INPUT(nn), 0, j + BITS) = (y >> j) & 1;
      }

      nn_forward(nn); // Move forward pass **after** setting inputs
      if (MAT_AT(NN_OUTPUT(nn), 0, BITS) > 0.5f) {
        printf("OVERFLOW\n");
      } else {
        size_t z = 0;
        for (size_t j = 0; j < BITS; j++) {
          size_t bit = MAT_AT(NN_OUTPUT(nn), 0, j) > 0.5f;
          z |= bit << j;
        }
        printf("%zu\n", z);
      }
    }
  }

  return 0;
}
