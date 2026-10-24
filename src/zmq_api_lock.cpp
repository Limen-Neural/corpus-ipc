// SPDX-License-Identifier: MIT OR Apache-2.0
//
// libzmq 4.3.4 segfaults in ypipe_t<command_t>::write when two threads enter
// zmq_connect (or socket create/close) at once. Ubuntu 26.04 runners hit that
// in `cargo test --all-features`. The command mailbox is supposed to allow one
// writer; concurrent C API calls still corrupt it. Send and recv stay
// unlocked so a blocking publish cannot stall the thread that drains it.
//
// Linked with -Wl,--wrap so the Rust `zmq` crate picks these up without a
// patch to that crate. The mutex is recursive because libzmq sometimes calls
// one wrapped entry point from another.

#include <pthread.h>
#include <stddef.h>

static pthread_mutex_t api_lock;
static pthread_once_t api_lock_once = PTHREAD_ONCE_INIT;

static void init_api_lock(void) {
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&api_lock, &attr);
    pthread_mutexattr_destroy(&attr);
}

struct ApiGuard {
    ApiGuard() {
        pthread_once(&api_lock_once, init_api_lock);
        pthread_mutex_lock(&api_lock);
    }
    ~ApiGuard() { pthread_mutex_unlock(&api_lock); }
};

extern "C" {

void *__real_zmq_socket(void *ctx, int type);
void *__wrap_zmq_socket(void *ctx, int type) {
    ApiGuard guard;
    return __real_zmq_socket(ctx, type);
}

int __real_zmq_close(void *socket);
int __wrap_zmq_close(void *socket) {
    ApiGuard guard;
    return __real_zmq_close(socket);
}

int __real_zmq_connect(void *socket, const char *addr);
int __wrap_zmq_connect(void *socket, const char *addr) {
    ApiGuard guard;
    return __real_zmq_connect(socket, addr);
}

int __real_zmq_bind(void *socket, const char *addr);
int __wrap_zmq_bind(void *socket, const char *addr) {
    ApiGuard guard;
    return __real_zmq_bind(socket, addr);
}

int __real_zmq_unbind(void *socket, const char *addr);
int __wrap_zmq_unbind(void *socket, const char *addr) {
    ApiGuard guard;
    return __real_zmq_unbind(socket, addr);
}

int __real_zmq_disconnect(void *socket, const char *addr);
int __wrap_zmq_disconnect(void *socket, const char *addr) {
    ApiGuard guard;
    return __real_zmq_disconnect(socket, addr);
}

int __real_zmq_setsockopt(void *socket, int option, const void *value, size_t len);
int __wrap_zmq_setsockopt(void *socket, int option, const void *value, size_t len) {
    ApiGuard guard;
    return __real_zmq_setsockopt(socket, option, value, len);
}

int __real_zmq_getsockopt(void *socket, int option, void *value, size_t *len);
int __wrap_zmq_getsockopt(void *socket, int option, void *value, size_t *len) {
    ApiGuard guard;
    return __real_zmq_getsockopt(socket, option, value, len);
}

int __real_zmq_ctx_set(void *ctx, int option, int value);
int __wrap_zmq_ctx_set(void *ctx, int option, int value) {
    ApiGuard guard;
    return __real_zmq_ctx_set(ctx, option, value);
}

}
