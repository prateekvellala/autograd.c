#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

typedef struct Scalar Scalar;


void build_topo(Scalar* v, Scalar*** topo, int* topo_count, int* topo_capacity,
                Scalar*** visited, int* visited_count, int* visited_capacity);
                

struct Scalar {
    double data;
    double grad;
    void (*_backward)(Scalar*);
    Scalar** _prev;
    int _prev_count;
    char* _op;
};

Scalar* create_scalar(double data, Scalar** children, int child_count, const char* op) {
    Scalar* s = malloc(sizeof(Scalar));
    s->data = data;
    s->grad = 0;
    s->_backward = NULL;
    s->_prev = NULL;
    s->_prev_count = 0;
    s->_op = NULL;

    if (child_count > 0 && children != NULL) {
        s->_prev = malloc(sizeof(Scalar*) * child_count);
        memcpy(s->_prev, children, sizeof(Scalar*) * child_count);
        s->_prev_count = child_count;
    }

    if (op != NULL) {
        s->_op = strdup(op);
    }

    return s;
}

void scalar_backward(Scalar* s);

void backward_add(Scalar* s) {
    Scalar* a = s->_prev[0];
    Scalar* b = s->_prev[1];
    a->grad += s->grad;
    b->grad += s->grad;
}

Scalar* scalar_add(Scalar* a, Scalar* b) {
    Scalar** children = malloc(2 * sizeof(Scalar*));
    children[0] = a;
    children[1] = b;
    Scalar* out = create_scalar(a->data + b->data, children, 2, "+");
    out->_backward = backward_add;
    return out;
}

void backward_mul(Scalar* s) {
    Scalar* a = s->_prev[0];
    Scalar* b = s->_prev[1];
    a->grad += b->data * s->grad;
    b->grad += a->data * s->grad;
}

Scalar* scalar_mul(Scalar* a, Scalar* b) {
    Scalar* out = create_scalar(a->data * b->data, (Scalar*[]){a, b}, 2, "*");
    out->_backward = backward_mul;
    return out;
}

void backward_pow(Scalar* s) {
    Scalar* a = s->_prev[0];
    double n = s->_op[1] - '0';
    a->grad += n * pow(a->data, n-1) * s->grad;
}

Scalar* scalar_pow(Scalar* a, double n) {
    char op[3] = "^0";
    op[1] = (char)(n + '0');
    Scalar* out = create_scalar(pow(a->data, n), (Scalar*[]){a}, 1, op);
    out->_backward = backward_pow;
    return out;
}

void backward_relu(Scalar* s) {
    Scalar* a = s->_prev[0];
    a->grad += (s->data > 0) * s->grad;
}

Scalar* scalar_relu(Scalar* a) {
    Scalar* out = create_scalar(a->data < 0 ? 0 : a->data, (Scalar*[]){a}, 1, "ReLU");
    out->_backward = backward_relu;
    return out;
}

void build_topo(Scalar* v, Scalar*** topo, int* topo_count, int* topo_capacity,
                Scalar*** visited, int* visited_count, int* visited_capacity) {
    printf("Entering build_topo for scalar with data %f\n", v->data);
    
    
    if (*visited_count >= *visited_capacity) {
        *visited_capacity *= 2;
        printf("Reallocating visited array to size %d\n", *visited_capacity);
        *visited = realloc(*visited, sizeof(Scalar*) * (*visited_capacity));
        if (!*visited) {
            fprintf(stderr, "Memory reallocation failed for visited\n");
            exit(1);
        }
    }


    for (int i = 0; i < *visited_count; i++) {
        if ((*visited)[i] == v) {
            printf("Scalar already visited, returning\n");
            return;
        }
    }

    (*visited)[(*visited_count)++] = v;
    printf("Added scalar to visited, count is now %d\n", *visited_count);

    printf("Scalar has %d children\n", v->_prev_count);
    for (int i = 0; i < v->_prev_count; i++) {
        printf("Processing child %d\n", i);
        if (v->_prev[i] == NULL) {
            printf("Child %d is NULL\n", i);
            continue;
        }
        build_topo(v->_prev[i], topo, topo_count, topo_capacity, visited, visited_count, visited_capacity);
    }

    if (*topo_count >= *topo_capacity) {
        *topo_capacity *= 2;
        printf("Reallocating topo array to size %d\n", *topo_capacity);
        *topo = realloc(*topo, sizeof(Scalar*) * (*topo_capacity));
        if (!*topo) {
            fprintf(stderr, "Memory reallocation failed for topo\n");
            exit(1);
        }
    }

    (*topo)[(*topo_count)++] = v;
    printf("Added scalar to topo, count is now %d\n", *topo_count);
}

void scalar_backward(Scalar* s) {
    printf("\nEntering scalar_backward\n");
    Scalar** topo = NULL;
    int topo_count = 0;
    int topo_capacity = 10;
    Scalar** visited = NULL;
    int visited_count = 0;
    int visited_capacity = 10;

    topo = malloc(sizeof(Scalar*) * topo_capacity);
    visited = malloc(sizeof(Scalar*) * visited_capacity);

    if (!topo || !visited) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    printf("Calling build_topo\n");
    build_topo(s, &topo, &topo_count, &topo_capacity, &visited, &visited_count, &visited_capacity);

    printf("Topological sort completed, topo_count: %d\n", topo_count);

    s->grad = 1;
    for (int i = topo_count - 1; i >= 0; i--) {
        printf("Processing scalar at index %d\n", i);
        if (topo[i]->_backward) {
            topo[i]->_backward(topo[i]);
        } else {
            printf("No backward function for scalar at index %d\n", i);
        }
    }

    free(topo);
    free(visited);
    printf("Exiting scalar_backward\n");
}

void backward_neg(Scalar* s) {
    s->_prev[0]->grad -= s->grad;
}

Scalar* scalar_neg(Scalar* a) {
    Scalar* result = create_scalar(-a->data, &a, 1, "neg");
    result->_backward = backward_neg;
    return result;
}

Scalar* scalar_sub(Scalar* a, Scalar* b) {
    return scalar_add(a, scalar_neg(b));
}

void backward_div(Scalar* s) {
    Scalar* a = s->_prev[0];
    Scalar* b = s->_prev[1];
    a->grad += s->grad / b->data;
    b->grad += -s->grad * a->data / (b->data * b->data);
}

Scalar* scalar_div(Scalar* a, Scalar* b) {
    Scalar* out = create_scalar(a->data / b->data, (Scalar*[]){a, b}, 2, "/");
    out->_backward = backward_div;
    return out;
}

char* scalar_repr(Scalar* s) {
    char* repr = malloc(100);
    snprintf(repr, 100, "Scalar(data=%f, grad=%f)", s->data, s->grad);
    return repr;
}

int main() {
    printf("Test Case 1: \n");

    Scalar* x = create_scalar(14.0, NULL, 0, "");
    Scalar* two = create_scalar(2.0, NULL, 0, "");
    Scalar* z = scalar_add(scalar_add(scalar_mul(two, x), two), x);
    Scalar* q = scalar_add(scalar_relu(z), scalar_mul(z, x));
    Scalar* h = scalar_relu(scalar_mul(z, z));
    Scalar* y = scalar_add(scalar_add(h, q), scalar_mul(q, x));

    scalar_backward(y);

    printf("\nForward pass: y.data = %f\n", y->data);
    printf("Backward pass: x.grad = %f\n", x->grad);

    free(x);
    free(two);
    free(z);
    free(q);
    free(h);
    free(y);

    printf("\nTest Case 2: \n");

    Scalar* a = create_scalar(-12.0, NULL, 0, "");
    Scalar* b = create_scalar(6.0, NULL, 0, "");
    Scalar* c = scalar_add(a, b);
    Scalar* d = scalar_add(scalar_mul(a, b), scalar_pow(b, 3));
    c = scalar_add(c, scalar_add(c, create_scalar(1.0, NULL, 0, "")));
    c = scalar_add(scalar_add(c, create_scalar(1.0, NULL, 0, "")), scalar_add(c, scalar_neg(a)));
    d = scalar_add(d, scalar_add(scalar_mul(d, create_scalar(2.0, NULL, 0, "")), scalar_relu(scalar_add(b, a))));
    d = scalar_add(d, scalar_add(scalar_mul(create_scalar(3.0, NULL, 0, ""), d), scalar_relu(scalar_sub(b, a))));
    Scalar* e = scalar_sub(c, d);
    Scalar* f = scalar_pow(e, 2);
    Scalar* g = scalar_div(f, create_scalar(2.0, NULL, 0, ""));
    g = scalar_add(g, scalar_div(create_scalar(10.0, NULL, 0, ""), f));

    scalar_backward(g);

    printf("\nForward pass: g.data = %f\n", g->data);
    printf("Backward pass: a.grad = %f, b.grad = %f\n", a->grad, b->grad);

    free(a);
    free(b);
    free(c);
    free(d);
    free(e);
    free(f);
    free(g);

    return 0;
}