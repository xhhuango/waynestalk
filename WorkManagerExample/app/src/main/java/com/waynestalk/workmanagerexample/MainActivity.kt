package com.waynestalk.workmanagerexample

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.work.*
import com.waynestalk.workmanagerexample.databinding.ActivityMainBinding
import java.util.concurrent.TimeUnit

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.cancelAllWorks.setOnClickListener {
            WorkManager.getInstance(this).cancelAllWork()
        }

        binding.oneTimeWork.setOnClickListener {
            val workRequest = OneTimeWorkRequestBuilder<LogWorker>()
                .setInputData(workDataOf(LogWorker.MESSAGE to "This is an one time work"))
                .setInitialDelay(1, TimeUnit.SECONDS)
                .build()
            WorkManager.getInstance(this).enqueue(workRequest)
        }

        binding.periodicWork.setOnClickListener {
            val workRequest = PeriodicWorkRequestBuilder<LogWorker>(
                PeriodicWorkRequest.MIN_PERIODIC_INTERVAL_MILLIS, TimeUnit.MILLISECONDS,
                PeriodicWorkRequest.MIN_PERIODIC_FLEX_MILLIS, TimeUnit.MILLISECONDS,
            )
                .setInputData(workDataOf(LogWorker.MESSAGE to "This is a periodic work"))
                .setInitialDelay(1, TimeUnit.SECONDS)
                .build()
            WorkManager.getInstance(this).enqueue(workRequest)
        }

        binding.oneTimeWorkWithNetworkConstraint.setOnClickListener {
            val constraints = Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .build()
            val workRequest = OneTimeWorkRequestBuilder<LogWorker>()
                .setInputData(workDataOf(LogWorker.MESSAGE to "This is an one time work with network constraint"))
                .setConstraints(constraints)
                .build()
            WorkManager.getInstance(this).enqueue(workRequest)
        }

        binding.uniqueOneTimeWork.setOnClickListener {
            val workRequest = OneTimeWorkRequestBuilder<LogWorker>()
                .setInputData(workDataOf(LogWorker.MESSAGE to "This is an unique one time work"))
                .setInitialDelay(5, TimeUnit.SECONDS)
                .build()
            WorkManager.getInstance(this).enqueueUniqueWork(
                "unique_one_time",
                ExistingWorkPolicy.REPLACE,
                workRequest,
            )
        }

        binding.retryWork.setOnClickListener {
            val workRequest = OneTimeWorkRequestBuilder<RetryLogWork>()
                .setInputData(workDataOf(RetryLogWork.MESSAGE to "This is a retry work"))
                .setBackoffCriteria(
                    BackoffPolicy.LINEAR,
                    OneTimeWorkRequest.MIN_BACKOFF_MILLIS,
                    TimeUnit.MILLISECONDS,
                )
                .build()
            WorkManager.getInstance(this).enqueue(workRequest);
        }

        binding.chainMultipleWorks.setOnClickListener {
            val workRequest1 = OneTimeWorkRequestBuilder<LogWorker>()
                .setInputData(workDataOf(LogWorker.MESSAGE to "Work 1"))
                .build()
            val workRequest2 = OneTimeWorkRequestBuilder<LogWorker>()
                .setInputData(workDataOf(LogWorker.MESSAGE to "Work 2"))
                .build()
            val workRequest3 = OneTimeWorkRequestBuilder<LogWorker>()
                .setInputData(workDataOf(LogWorker.MESSAGE to "Work 3"))
                .build()
            WorkManager.getInstance(this)
                .beginWith(listOf(workRequest1, workRequest2))
                .then(workRequest3)
                .enqueue()
        }
    }
}