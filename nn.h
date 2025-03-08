#ifndef NN_H_
#define NN_H_
// Header file implementation
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

float rand_float(void);  // rand matrix initiation
float sigmoidf(float x); // sigmoid activation
typedef struct {
  size_t rows;
  size_t cols;
  size_t stride;
  float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i) * (m).stride + (j)]

Mat mat_alloc(size_t row, size_t col);       // matrix allocation
void mat_rand(Mat m, float high, float low); // Iniatialise the random matrix
void mat_dot(Mat dst, Mat a, Mat b);         // Matrix dot product
void mat_sum(Mat dst, Mat a);                // Matrix sum
void mat_fill(Mat m, float x);               // Fill matrix with float x
void mat_free(Mat m);            // Frees the memory allocated for the matrix.1
void mat_sig(Mat m);             // element wise sigmoid of matrix
Mat mat_row(Mat m, size_t row);  // row of matrix as row vector
void mat_copy(Mat dst, Mat src); // copy mat src to matrix dst
void mat_print(Mat m, const char *name, size_t padding); // print matrix
#define MAT_PRINT(m)                                                           \
  mat_print(m, #m, 0) // #m strifies it, simplifying matrix print
void mat_free(Mat m);

typedef struct {
  size_t count; // number of layers
  Mat *ws;      // pointer to weights
  Mat *bs;      // pointer to biases
  Mat *as;      // pointer to activations, the amount of activation is count + 1

} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_zero(NN nn);
void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn)
void nn_rand(NN nn, float high,
             float low); // Iniatialise the random Neural Network
void nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to);
void nn_backprop(NN nn, NN g, Mat ti, Mat to);
void nn_learn(NN nn, NN g, float rate);
void nn_free(NN nn);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION
float rand_float(void) { return (float)rand() / (float)RAND_MAX; }
Mat mat_alloc(size_t rows, size_t cols) {
  Mat m;
  m.rows = rows;
  m.cols = cols;
  m.stride = cols;
  m.es = malloc(sizeof(*m.es) * rows * cols);
  assert(m.es != NULL);
  return m;
}

// Frees the memory allocated for the matrix.
void mat_free(Mat m) { free(m.es); }

void mat_print(Mat m, const char *name, size_t padding) {
  printf("%*s%s = [\n", (int)padding, "", name);
  for (size_t i = 0; i < m.rows; ++i) {
    printf("%*s    ", (int)padding, "");
    for (size_t j = 0; j < m.cols; ++j) {
      printf("%f ", MAT_AT(m, i, j));
    }
    printf("\n");
  }
  printf("%*s]\n", (int)padding, "");
}

void mat_rand(Mat m, float high, float low) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) = rand_float() * (high - low) + low; // low < high
    }
  }
}
float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

void mat_sig(Mat m) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
    }
  }
}
Mat mat_row(Mat m, size_t row) {
  return (Mat){
      .rows = 1,
      .cols = m.cols,
      .stride = m.stride,
      .es = &MAT_AT(m, row, 0),
  };
}
void mat_copy(Mat dst, Mat src) {
  assert(dst.rows == src.rows);
  assert(dst.cols == src.cols);
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) = MAT_AT(src, i, j);
    }
  }
}
void mat_sum(Mat dst, Mat a) {
  assert(dst.rows == a.rows);
  assert(dst.cols == a.cols);
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) += MAT_AT(a, i, j);
    }
  }
}

void mat_fill(Mat m, float x) {
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.cols; j++) {
      MAT_AT(m, i, j) = x;
    }
  }
}

void mat_dot(Mat dst, Mat a, Mat b) {
  assert(a.cols == b.rows);
  size_t n = a.cols;
  assert(dst.rows == a.rows);
  assert(dst.cols == b.cols);

  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) = 0;
      for (size_t k = 0; k < n; ++k) {
        MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
      }
    }
  }
}

// size_t arch[] = {2, 2, 1} // input number, 1 hidden layer-> 2 neurons, output
// layer -> 1 neuron NN nn = nn_alloc(arch, ARRAY_LEN(arch, ARRAY_LEN(arch)));
NN nn_alloc(size_t *arch, size_t arch_count) {
  NN nn;
  nn.count = arch_count - 1; // arch count takes input layer into account, but
                             // count is for hidden layer only
  nn.ws = malloc(sizeof(*nn.ws) * nn.count);
  assert(nn.ws != NULL);
  nn.bs = malloc(sizeof(*nn.bs) * nn.count);
  assert(nn.bs != NULL);
  // nn.as = malloc(sizeof(*nn.as) * (nn.count - 1));
  nn.as = malloc(sizeof(*nn.as) * (nn.count + 1));
  assert(nn.as != NULL);
  nn.as[0] = mat_alloc(1, arch[0]); // input layer
  for (size_t i = 1; i < arch_count; i++) {
    nn.ws[i - 1] = mat_alloc(nn.as[i - 1].cols, arch[i]);
    nn.bs[i - 1] = mat_alloc(1, arch[i]);
    nn.as[i] = mat_alloc(1, arch[i]);
  }

  return nn;
}

void nn_zero(NN nn) {
  for (size_t i = 0; i < nn.count; i++) {
    mat_fill(nn.ws[i], 0);
    mat_fill(nn.bs[i], 0);
    mat_fill(nn.as[i], 0);
  }
  mat_fill(nn.as[nn.count], 0);
}

void nn_print(NN nn, const char *name) {
  char buf[256];
  printf("%s = [\n", name);
  for (size_t i = 0; i < nn.count; ++i) {
    snprintf(buf, sizeof(buf), "ws%zu", i);
    mat_print(nn.ws[i], buf, 4);
    snprintf(buf, sizeof(buf), "bs%zu", i);
    mat_print(nn.bs[i], buf, 4);
  }
  printf("]\n");
}
void nn_rand(NN nn, float high, float low) {
  for (size_t i = 0; i < nn.count; ++i) {
    mat_rand(nn.ws[i], high, low);
    mat_rand(nn.bs[i], high, low);
  }
}

void nn_forward(NN nn) {
  for (size_t i = 0; i < nn.count; i++) {
    mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
    mat_sum(nn.as[i + 1], nn.bs[i]);
    mat_sig(nn.as[i + 1]);
  }
}

float nn_cost(NN nn, Mat ti, Mat to) {
  assert(ti.rows == to.rows);
  assert(to.cols == NN_OUTPUT(nn).cols);
  size_t n = ti.rows;
  float c = 0;

  for (size_t i = 0; i < n; i++) {
    Mat x = mat_row(ti, i); // Input
    Mat y = mat_row(to, i); // output

    mat_copy(NN_INPUT(nn), x);
    nn_forward(nn);
    NN_OUTPUT(nn);
    size_t q = to.cols;
    for (size_t j = 0; j < q; j++) {
      float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
      c += d * d;
    }
  }
  return c / n;
}

void nn_backprop(NN nn, NN g, Mat ti, Mat to) {
  assert(ti.rows == to.rows);
  size_t n = ti.rows;
  assert(NN_OUTPUT(nn).cols == to.cols);

  nn_zero(g); // clear out previous gardients
  // i - current sample
  // l - current layer
  // j - current activation
  // k - previous activation

  for (size_t i = 0; i < n; i++) {
    // for each input
    mat_copy(NN_INPUT(nn), mat_row(ti, i));
    nn_forward(nn);

    for (size_t j = 0; j < nn.count; j++) {
      mat_fill(g.as[j], 0);
    }
    for (size_t j = 0; j < to.cols; j++) {
      // for each output, 1 input vector might have multiple outputs
      MAT_AT(NN_OUTPUT(g), 0, j) =
          2 * (MAT_AT(NN_OUTPUT(nn), 0, j) -
               MAT_AT(to, i, j)); // store the partial derivatives in g
    }
    for (size_t l = nn.count; l > 0; l--) {
      // for each layer
      for (size_t j = 0; j < nn.as[l].cols; j++) {
        // b can be computed here only
        float a = MAT_AT(nn.as[l], 0, j);
        float da = MAT_AT(g.as[l], 0, j);
        MAT_AT(g.bs[l - 1], 0, j) += a * (1 - a) * da;
        for (size_t k = 0; k < nn.as[l - 1].cols; k++) {
          // j - weight matrix col
          // k - weight matrix row
          float pa = MAT_AT(nn.as[l - 1], 0, k); // activation of previous layer
          float w = MAT_AT(nn.ws[l - 1], k, j);  // weight of previous layer
          MAT_AT(g.ws[l - 1], k, j) += a * da * (1 - a) * pa;
          MAT_AT(g.as[l - 1], 0, k) += a * da * (1 - a) * w;
        }
      }
    }
  }

  for (size_t i = 0; i < g.count; i++) {
    for (size_t j = 0; j < g.ws[i].rows; j++) {
      for (size_t k = 0; k < g.ws[i].cols; k++) {
        MAT_AT(g.ws[i], j, k) /= n;
      }
    }
    for (size_t j = 0; j < g.bs[i].rows; j++) {
      for (size_t k = 0; k < g.bs[i].cols; k++) {
        MAT_AT(g.bs[i], j, k) /= n;
      }
    }
  }
}

void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to) {
  float saved;
  float c = nn_cost(nn, ti, to);
  // layer
  // matrix(i,j)
  for (size_t i = 0; i < nn.count; i++) {
    for (size_t j = 0; j < nn.ws[i].rows; j++) {
      for (size_t k = 0; k < nn.ws[i].cols; k++) {
        saved = MAT_AT(nn.ws[i], j, k); // save old value
        MAT_AT(nn.ws[i], j, k) += eps;  // wiggle
        MAT_AT(g.ws[i], j, k) =
            (nn_cost(nn, ti, to) - c) / eps; // store the grad in g
        MAT_AT(nn.ws[i], j, k) = saved;      // save the slightly changed value
      }
    }
    for (size_t j = 0; j < nn.bs[i].rows; j++) {
      for (size_t k = 0; k < nn.bs[i].cols; k++) {
        saved = MAT_AT(nn.bs[i], j, k); // save old value
        MAT_AT(nn.bs[i], j, k) += eps;  // wiggle
        MAT_AT(g.bs[i], j, k) =
            (nn_cost(nn, ti, to) - c) / eps; // store the grad in g
        MAT_AT(nn.bs[i], j, k) = saved;      // save the slightly changed value
      }
    }
  }
}

void nn_learn(NN nn, NN g, float rate) {
  for (size_t i = 0; i < nn.count; i++) {
    for (size_t j = 0; j < nn.ws[i].rows; j++) {
      for (size_t k = 0; k < nn.ws[i].cols; k++) {
        MAT_AT(nn.ws[i], j, k) -= rate * MAT_AT(g.ws[i], j, k);
      }
    }
    for (size_t j = 0; j < nn.bs[i].rows; j++) {
      for (size_t k = 0; k < nn.bs[i].cols; k++) {
        MAT_AT(nn.bs[i], j, k) -= rate * MAT_AT(g.bs[i], j, k);
      }
    }
  }
}
void nn_free(NN nn) {
  // Free weight matrices and bias matrices
  for (size_t i = 0; i < nn.count; i++) {
    mat_free(nn.ws[i]); // Free each weight matrix
    mat_free(nn.bs[i]); // Free each bias matrix
  }
  // Free the activations array
  for (size_t i = 0; i < nn.count + 1; i++) {
    mat_free(nn.as[i]); // Free each activation matrix
  }
  // Free the arrays that hold the matrices
  free(nn.ws);
  free(nn.bs);
  free(nn.as);
}

#endif // NN_IMPLEMENTATION