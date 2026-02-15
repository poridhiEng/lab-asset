# Atomic Operations

## Introduction

In the previous lab, we fixed data races using `sync.Mutex`. Mutexes work correctly, but they carry overhead: acquiring a lock, releasing it, and potentially blocking other goroutines while one holds the lock. For simple operations like incrementing a counter or setting a flag, this overhead is unnecessary. Go provides a lighter-weight tool: the `sync/atomic` package.

### What Are Atomic Operations?

An atomic operation completes in a single, indivisible step from the perspective of other goroutines. No goroutine can observe the operation half-done. When one goroutine atomically increments a counter, another goroutine will either see the value before the increment or after, never a corrupted intermediate state. The CPU guarantees this at the hardware level using special instructions.

![](./images/5.svg)

### How Atomics Work at the CPU Level

To understand why atomics are faster, let's look at what `counter++` actually does. The CPU breaks it into three steps:

1. **Load** the current value from memory into a CPU register
2. **Add** 1 to the value in the register
3. **Store** the result back to memory

When two goroutines run on two CPU cores, both can load the same value before either stores, causing a lost update. A mutex solves this by wrapping all three steps in a lock, but that involves multiple operations itself: acquiring the lock (which is internally an atomic CAS), doing the work, releasing the lock, and potentially putting goroutines to sleep and waking them via the OS scheduler.

An atomic operation solves this at the hardware level. Modern CPUs provide special instructions (like `LOCK XADD` on x86 or `LDXR/STXR` on ARM) that perform the entire read-modify-write as a single indivisible instruction. The CPU's cache coherence protocol ensures that when one core executes an atomic instruction, all other cores see the update immediately. No software lock, no goroutine blocking, no scheduler involvement.

The difference in practice:

| | Mutex path | Atomic path |
| --- | --- | --- |
| Steps | Lock → Read → Modify → Write → Unlock | Single CPU instruction |
| Blocking | Other goroutines wait (sleep/wake) | No goroutine ever blocks |
| Scheduler | May involve OS scheduler | No scheduler involvement |
| Best for | Multi-step logic, multiple variables | Single variable, single operation |

### When to Use Atomic vs Mutex

| Use Case | Tool | Why |
| --- | --- | --- |
| Simple counter (increment/decrement) | `atomic` | Single variable, single operation |
| Read or write a single value | `atomic` | No multi-step logic needed |
| Check balance then deduct | `sync.Mutex` | Multiple variables or multi-step logic |
| Update several related fields together | `sync.Mutex` | Need to keep fields consistent |

**Rule of thumb:** If you're protecting a single variable with a single operation, use atomic. If you need to protect multiple variables or a sequence of steps, use a mutex.

### Available Operations

The `sync/atomic` package provides five operations across several integer types:

| Operation | What it does | Supported types |
| --- | --- | --- |
| `Load` | Read a value atomically | `int32`, `int64`, `uint32`, `uint64`, `uintptr`, `Pointer` |
| `Store` | Write a value atomically | `int32`, `int64`, `uint32`, `uint64`, `uintptr`, `Pointer` |
| `Add` | Add and return new value | `int32`, `int64`, `uint32`, `uint64`, `uintptr` |
| `Swap` | Replace and return old value | `int32`, `int64`, `uint32`, `uint64`, `uintptr`, `Pointer` |
| `CompareAndSwap` | Replace only if current == old | `int32`, `int64`, `uint32`, `uint64`, `uintptr`, `Pointer` |

The function names follow the pattern `Operation` + `Type`, for example `atomic.AddInt64`, `atomic.LoadUint32`, `atomic.CompareAndSwapInt32`.

For complex types (structs, slices, maps), use `atomic.Value` which stores and loads `interface{}` values atomically.

## Task Description

In this lab, we will:

- Replace a mutex-protected counter with `atomic.AddInt64` and observe correctness
- Use `atomic.LoadInt64` and `atomic.StoreInt64` for safe reads and writes
- Use `atomic.SwapInt64` to atomically replace a value and get the old one
- Implement a lock-free pattern with `atomic.CompareAndSwapInt64` (CAS)
- Use the `atomic.Value` type for atomically storing and loading complex types
- Build a practical example: concurrent metrics collector using atomic operations

## Atomic Operations Fundamentals

### 1. From Mutex to Atomic

![](./images/3.svg)

In Lab 01, we fixed the shared counter with a mutex:

```go
mu.Lock()
counter++
mu.Unlock()
```

This works, but every goroutine must acquire and release the lock, even for a single `counter++`. With `sync/atomic`, we can do the same thing without any locking.

Create a file named `main.go` and add the following code:

```go
package main

import (
    "fmt"
    "sync"
    "sync/atomic"
)

func main() {
    var counter int64 // must be int64, not int
    var wg sync.WaitGroup

    for i := 0; i < 1000; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            atomic.AddInt64(&counter, 1) // atomic increment
        }()
    }

    wg.Wait()
    fmt.Printf("Expected: 1000, Got: %d\n", counter)
}
```

`atomic.AddInt64(&counter, 1)` atomically reads, adds 1, and writes the result back in a single CPU instruction. No lock is needed. Note that the variable must be `int64` (not `int`), because atomic functions operate on fixed-size types.

Run the program:

```bash
go run -race main.go
```

**Expected Output:**

```txt
Expected: 1000, Got: 1000
```

No race warnings, always exactly 1000. To decrement, use a negative value:

```go
atomic.AddInt64(&counter, -1)
```

### 2. Load and Store

Reading or writing a regular variable while another goroutine modifies it is a data race, even for a single assignment like `flag = true`. On some architectures, a 64-bit write isn't atomic (the CPU writes it in two 32-bit halves), so a concurrent read can see a half-written value.

![](./images/2.svg)

`atomic.LoadInt64` and `atomic.StoreInt64` guarantee that reads and writes are atomic on all platforms.

Update `main.go`:

```go
package main

import (
    "fmt"
    "sync/atomic"
    "time"
)

func main() {
    var running int64 = 1 // 1 = running, 0 = stopped

    // Worker checks the flag in a loop
    go func() {
        for atomic.LoadInt64(&running) == 1 {
            fmt.Println("working...")
            time.Sleep(100 * time.Millisecond)
        }
        fmt.Println("worker stopped")
    }()

    // Let the worker run for a bit, then signal shutdown
    time.Sleep(350 * time.Millisecond)
    atomic.StoreInt64(&running, 0) // signal: stop
    time.Sleep(150 * time.Millisecond) // wait for worker to see it

    fmt.Println("main exiting")
}
```

The worker goroutine reads `running` with `atomic.LoadInt64` on every iteration. The main goroutine writes `0` with `atomic.StoreInt64` to signal shutdown. Both operations are atomic, so no mutex is needed and there is no data race.

Run the program:

```bash
go run -race main.go
```

**Expected Output:**

```txt
working...
working...
working...
worker stopped
main exiting
```

This is a common pattern for shutdown flags. The worker doesn't need to acquire a lock on every loop iteration. `LoadInt64` is essentially free compared to `mu.Lock()`.

### 3. Swap

`atomic.SwapInt64` atomically sets a new value and returns the old one. This is useful when you need to "reset" a counter and capture what it was. For example, collecting metrics over intervals.

![](./images/1.svg)

Update `main.go`:

```go
package main

import (
    "fmt"
    "sync"
    "sync/atomic"
    "time"
)

func main() {
    var requestCount int64

    // Simulate incoming requests (10 goroutines x 50 requests, 10ms apart)
    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for j := 0; j < 50; j++ {
                atomic.AddInt64(&requestCount, 1)
                time.Sleep(10 * time.Millisecond)
            }
        }()
    }

    // Periodic reporting: reset counter and report every 100ms
    reportDone := make(chan struct{})
    go func() {
        for i := 0; i < 5; i++ {
            time.Sleep(100 * time.Millisecond)
            count := atomic.SwapInt64(&requestCount, 0) // reset to 0, get old value
            fmt.Printf("Interval %d: %d requests\n", i+1, count)
        }
        close(reportDone)
    }()

    wg.Wait()
    <-reportDone // wait for reporter to finish

    // Drain any remaining requests after the last report
    remaining := atomic.SwapInt64(&requestCount, 0)
    if remaining > 0 {
        fmt.Printf("Remaining:  %d requests\n", remaining)
    }
}
```

`atomic.SwapInt64(&requestCount, 0)` atomically sets `requestCount` to 0 and returns whatever it was before the swap. No requests are lost. Every increment is counted in exactly one interval. Without `Swap`, we'd need a mutex to read-then-reset the counter, and we'd risk missing increments that happen between the read and the reset.

Each worker goroutine runs for ~500ms (50 iterations x 10ms). The reporter fires every 100ms, so it captures 5 intervals of live traffic. Main waits for both the workers and the reporter to finish before draining any leftover count.

Run the program:

```bash
go run -race main.go
```

**Expected Output (varies):**

```txt
Interval 1: 100 requests
Interval 2: 80 requests
Interval 3: 79 requests
Interval 4: 90 requests
Interval 5: 100 requests
Remaining:  51 requests
```

The exact counts per interval vary, but the sum is always 500 (10 goroutines x 50 requests).

### 4. CompareAndSwap (CAS)

`atomic.CompareAndSwapInt64` (CAS) is the most powerful atomic operation. It says: "If the current value is `old`, replace it with `new` and return `true`. Otherwise, do nothing and return `false`."

![](./images/4.svg)

```go
swapped := atomic.CompareAndSwapInt64(&val, old, new)
```

CAS is the building block of most lock-free algorithms. It lets you implement conditional updates without a mutex.

#### One-Time Initialization

A common use case: ensure something happens exactly once, without `sync.Once` or a mutex.

Update `main.go`:

```go
package main

import (
    "fmt"
    "sync"
    "sync/atomic"
)

func main() {
    var initialized int64 // 0 = not yet, 1 = done
    var wg sync.WaitGroup

    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            // Only the first goroutine to CAS from 0 → 1 succeeds
            if atomic.CompareAndSwapInt64(&initialized, 0, 1) {
                fmt.Printf("Goroutine %d: I initialized the resource!\n", id)
            } else {
                fmt.Printf("Goroutine %d: already initialized, skipping\n", id)
            }
        }(i)
    }

    wg.Wait()
}
```

All 10 goroutines try to CAS `initialized` from 0 to 1. Exactly one succeeds (the first one to execute the CAS). The rest see that the value is already 1 (not 0), so CAS returns `false` and they skip initialization.

Run the program:

```bash
go run -race main.go
```

**Expected Output (varies):**

```txt
Goroutine 3: I initialized the resource!
Goroutine 0: already initialized, skipping
Goroutine 1: already initialized, skipping
Goroutine 2: already initialized, skipping
Goroutine 4: already initialized, skipping
Goroutine 5: already initialized, skipping
Goroutine 6: already initialized, skipping
Goroutine 7: already initialized, skipping
Goroutine 8: already initialized, skipping
Goroutine 9: already initialized, skipping
```

Exactly one goroutine prints "I initialized". The winner varies, but there's always exactly one.

#### CAS Loop

When you need to read a value, compute something, and write it back without a mutex, you use a CAS loop. The loop reads the current value, computes the new value, and attempts to CAS. If another goroutine changed the value in between, CAS fails and the loop retries with the updated value.

Update `main.go`:

```go
package main

import (
    "fmt"
    "sync"
    "sync/atomic"
)

// atomicMax updates max to be the larger of max and val, without a mutex.
func atomicMax(max *int64, val int64) {
    for {
        old := atomic.LoadInt64(max)
        if val <= old {
            return // current max is already >= val
        }
        if atomic.CompareAndSwapInt64(max, old, val) {
            return // successfully updated
        }
        // CAS failed: another goroutine changed max. Retry.
    }
}

func main() {
    var max int64
    var wg sync.WaitGroup

    values := []int64{42, 17, 99, 55, 88, 73, 100, 61, 3, 95}

    for _, v := range values {
        wg.Add(1)
        go func(val int64) {
            defer wg.Done()
            atomicMax(&max, val)
        }(v)
    }

    wg.Wait()
    fmt.Printf("Max value: %d\n", atomic.LoadInt64(&max))
}
```

The `atomicMax` function loads the current max, checks if the new value is larger, and attempts to CAS. If CAS fails (another goroutine updated `max` between our load and CAS), we retry. The loop always terminates because either our value isn't the max (return immediately) or we eventually succeed with CAS.

Run the program:

```bash
go run -race main.go
```

**Expected Output:**

```txt
Max value: 100
```

No mutex, no blocking, just atomic operations. This pattern works well for counters, max/min trackers, and other single-variable computations.

### 5. atomic.Value

The atomic functions we've seen so far work with integers. But what if you need to atomically store and load a struct, a slice, or any complex type? That's what `atomic.Value` is for.

`atomic.Value` stores an `interface{}`. You can atomically `Store` any value and `Load` it back. Multiple goroutines can read concurrently while one goroutine updates the value, all without a mutex.

Update `main.go`:

```go
package main

import (
    "fmt"
    "sync/atomic"
    "time"
)

type Config struct {
    MaxConns    int
    Timeout     time.Duration
    DebugMode   bool
}

func main() {
    var config atomic.Value

    // Set initial config
    config.Store(Config{
        MaxConns:  10,
        Timeout:   5 * time.Second,
        DebugMode: false,
    })

    // Reader goroutines: access config without locking
    for i := 0; i < 3; i++ {
        go func(id int) {
            for j := 0; j < 3; j++ {
                cfg := config.Load().(Config)
                fmt.Printf("Reader %d: MaxConns=%d, Debug=%v\n", id, cfg.MaxConns, cfg.DebugMode)
                time.Sleep(50 * time.Millisecond)
            }
        }(i)
    }

    // Writer goroutine — updates config periodically
    time.Sleep(80 * time.Millisecond)
    fmt.Println("--- Updating config ---")
    config.Store(Config{
        MaxConns:  50,
        Timeout:   10 * time.Second,
        DebugMode: true,
    })

    time.Sleep(200 * time.Millisecond)
}
```

Readers call `config.Load().(Config)` to get the current config. The type assertion `.(Config)` converts the `interface{}` back to a `Config` struct. The writer calls `config.Store(...)` to replace the entire config atomically. Readers always see either the old config or the new one, never a partial update.

Run the program:

```bash
go run -race main.go
```

**Expected Output (varies):**

```txt
Reader 0: MaxConns=10, Debug=false
Reader 1: MaxConns=10, Debug=false
Reader 2: MaxConns=10, Debug=false
--- Updating config ---
Reader 0: MaxConns=50, Debug=true
Reader 1: MaxConns=50, Debug=true
Reader 2: MaxConns=50, Debug=true
Reader 0: MaxConns=50, Debug=true
Reader 1: MaxConns=50, Debug=true
Reader 2: MaxConns=50, Debug=true
```

Before the update, all readers see `MaxConns=10`. After the update, they all see `MaxConns=50`. No reader ever sees `MaxConns=50` with `Debug=false` because the struct is replaced atomically as a whole.

**Important:** Always store the same concrete type in an `atomic.Value`. Storing a `Config` and then a `string` will panic at runtime.

## Conclusion

In this lab, we learned the `sync/atomic` package for lightweight concurrent state management.
- `AddInt64` handles counters without locking
- `Load`/`Store` provide safe reads and writes for flags
- `Swap` enables atomic reset-and-capture
- `CompareAndSwap` (CAS) powers lock-free conditional updates
- `atomic.Value` provides atomic load and store for complex types like structs

Atomic operations are faster than mutexes because they map directly to single CPU instructions with no goroutine blocking or scheduler involvement. But mutexes remain the right choice when you need to protect multiple variables or multi-step logic together. In the next lab, we'll learn debugging and profiling tools: `pprof` for CPU/memory profiling, goroutine leak detection, and trace analysis.
