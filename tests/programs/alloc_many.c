#include <stdlib.h>
int init_random(int *a, int *b) {
  for (int i = 0; i < 256 * 1024; i++) {
    a[i] = rand();
    b[i] = rand();
  }
}
int main() {
  int a[256 * 1024] = {};
  int b[256 * 1024] = {};

  for (int i = 0; i < 256 * 1024; i++) {
    if (rand() > 100)
      a[i] = a[i + 1] * rand() + b[i];
    else
      a[i] = a[i - 1] * rand() + b[i];
  }
  return 0;
}
