package com.waynestalk.androidrecyclerviewmultipleitemsexample

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val recyclerView = findViewById<RecyclerView>(R.id.mainRecyclerView)
        recyclerView.adapter = RowAdapter(
            listOf(
                TextRowData("Name", "Wayne's Take"),
                EditRowData("Phone Number", "12345678") { println("Phone number: $it") },
                SwitchRowData("Registered", false) { println("Registered: $it") }
            )
        )
        recyclerView.layoutManager = LinearLayoutManager(this)
    }
}