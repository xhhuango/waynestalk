import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.onEach

fun main(args: Array<String>) {
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

fun runSharedFlowWithoutCache() {
    val sharedFlow = MutableSharedFlow<Int>()

    GlobalScope.launch(Dispatchers.IO) {
        sharedFlow.emit(-3)
        sharedFlow.emit(-2)
        sharedFlow.emit(-1)
    }

    val job1 = sharedFlow
        .onEach {
            println("#1 got $it (${currentCoroutineContext()})")
        }
        .launchIn(GlobalScope)

    val job2 = sharedFlow
        .onEach {
            println("#2 got $it (${currentCoroutineContext()})")
        }
        .launchIn(GlobalScope)

    val job = GlobalScope.launch(Dispatchers.IO) {
        repeat(100) {
            println("Emitting $it (${currentCoroutineContext()})")
            sharedFlow.emit(it)
            delay(1000)
        }
    }

    println("Enter any key to cancel job1")
    readln()
    job1.cancel()

    println("Enter any key to cancel job2")
    readln()
    job2.cancel()

    println("Enter any key to stop")
    readln()
    job.cancel()
}

fun runSharedFlowWithCache() {
    val sharedFlow = MutableSharedFlow<Int>(replay = 2)

    GlobalScope.launch(Dispatchers.IO) {
        sharedFlow.emit(-3)
        sharedFlow.emit(-2)
        sharedFlow.emit(-1)
    }

    val job1 = sharedFlow
        .onEach {
            println("#1 got $it (${currentCoroutineContext()})")
        }
        .launchIn(GlobalScope)

    val job2 = sharedFlow
        .onEach {
            println("#2 got $it (${currentCoroutineContext()})")
        }
        .launchIn(GlobalScope)

    val job = GlobalScope.launch(Dispatchers.IO) {
        repeat(100) {
            println("Emitting $it (${currentCoroutineContext()})")
            sharedFlow.emit(it)
            delay(1000)
        }
    }

    println("Enter any key to cancel job1")
    readln()
    job1.cancel()

    println("Enter any key to cancel job2")
    readln()
    job2.cancel()

    println("Enter any key to stop")
    readln()
    job.cancel()
}

fun runStateFlow() {
    val stateFlow = MutableStateFlow(-100)

    val job1 = stateFlow
        .onEach {
            println("#1 got $it (${currentCoroutineContext()})")
        }
        .launchIn(GlobalScope)

    val job2 = stateFlow
        .onEach {
            println("#2 got $it (${currentCoroutineContext()})")
        }
        .launchIn(GlobalScope)

    val job = GlobalScope.launch(Dispatchers.IO) {
        delay(1000)
        repeat(100) {
            println("Emitting $it (${currentCoroutineContext()})")
            stateFlow.emit(it)
            delay(1000)
        }
    }

    println("Enter any key to cancel job1")
    readln()
    job1.cancel()

    println("Enter any key to cancel job2")
    readln()
    job2.cancel()

    println("Enter any key to stop")
    readln()
    job.cancel()
}