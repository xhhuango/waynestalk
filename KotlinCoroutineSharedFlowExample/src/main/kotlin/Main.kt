import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.onEach

val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

suspend fun main(args: Array<String>) {
    while (true) {
        println("Enter:")
        println("  - 1 to run demo of SharedFlow without cache")
        println("  - 2 to run demo of ShredFlow with cache")
        println("  - 3 to run demo of StateFlow with cache")
        println("  - exit to terminate the program")
        when (readLine()) {
            "1" -> runSharedFlowWithoutCache()
            "2" -> runSharedFlowWithCache()
            "3" -> runStateFlow()
            "exit" -> return
            else -> continue
        }
    }
}

suspend fun runSharedFlowWithoutCache() {
    val sharedFlow = MutableSharedFlow<Int>()

    sharedFlow.emit(-3)
    sharedFlow.emit(-2)
    sharedFlow.emit(-1)

    val subscriber1 = sharedFlow
        .onEach {
            println("#1 got $it (${currentCoroutineContext()})")
        }
        .launchIn(scope)

    val subscriber2 = sharedFlow
        .onEach {
            println("#2 got $it (${currentCoroutineContext()})")
        }
        .launchIn(scope)

    val publisher2 = scope.launch {
        repeat(100) {
            println("Emitting $it (${currentCoroutineContext()})")
            sharedFlow.emit(it)
            delay(1000)
        }
    }

    println("Enter any key to cancel subscriber1")
    readln()
    subscriber1.cancel()

    println("Enter any key to cancel subscriber2")
    readln()
    subscriber2.cancel()

    println("Enter any key to stop emitting values")
    readln()
    publisher2.cancel()
}

suspend fun runSharedFlowWithCache() {
    val sharedFlow = MutableSharedFlow<Int>(replay = 2)

    sharedFlow.emit(-3)
    sharedFlow.emit(-2)
    sharedFlow.emit(-1)

    val subscribe1 = sharedFlow
        .onEach {
            println("#1 got $it (${currentCoroutineContext()})")
        }
        .launchIn(scope)

    val subscribe2 = sharedFlow
        .onEach {
            println("#2 got $it (${currentCoroutineContext()})")
        }
        .launchIn(scope)

    val publisher1 = scope.launch {
        repeat(100) {
            println("Emitting $it (${currentCoroutineContext()})")
            sharedFlow.emit(it)
            delay(1000)
        }
    }

    println("Enter any key to cancel subscribe1")
    readln()
    subscribe1.cancel()

    println("Enter any key to cancel subscribe2")
    readln()
    subscribe2.cancel()

    println("Enter any key to stop emitting values")
    readln()
    publisher1.cancel()
}

suspend fun runStateFlow() {
    val stateFlow = MutableStateFlow(-100)

    val subscriber1 = stateFlow
        .onEach {
            println("#1 got $it (${currentCoroutineContext()})")
        }
        .launchIn(scope)

    val subscriber2 = stateFlow
        .onEach {
            println("#2 got $it (${currentCoroutineContext()})")
        }
        .launchIn(scope)

    val job = scope.launch {
        delay(1000)
        repeat(100) {
            val value = it / 2
            println("Emitting $value (${currentCoroutineContext()})")
            stateFlow.emit(value)
            delay(1000)
        }
    }

    println("Enter any key to cancel subscriber1")
    readln()
    subscriber1.cancel()

    println("Enter any key to cancel subscriber2")
    readln()
    subscriber2.cancel()

    println("Enter any key to stop emitting values")
    readln()
    job.cancel()
}
