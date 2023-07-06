package com.waynestalk.collectionappwidget

import android.content.Context
import android.content.Intent
import android.widget.RemoteViews
import android.widget.RemoteViewsService

class PersonInfoAppWidgetService : RemoteViewsService() {
    override fun onGetViewFactory(intent: Intent?): RemoteViewsFactory {
        return PersonInfoRemoteViewFactory(this)
    }
}

class PersonInfoRemoteViewFactory(
    private val context: Context,
) : RemoteViewsService.RemoteViewsFactory {
    private lateinit var people: List<Person>

    override fun onCreate() {
        loadData()
    }

    override fun onDataSetChanged() {
        loadData()
    }

    private fun loadData() {
        people = listOf(
            Person("Wayne", "Mobile Software Developer & Blogger"),
            Person("David", "Android Developer"),
            Person("Peter", "Embedded System Developer"),
        )
    }

    override fun onDestroy() {
    }

    override fun getCount(): Int {
        print("count=${people.size}")
        return people.size
    }

    override fun getViewAt(position: Int): RemoteViews {
        return RemoteViews(context.packageName, R.layout.person_info_item).apply {
            val person = people[position]
            setTextViewText(R.id.name_text_view, person.name)
            setTextViewText(R.id.job_text_view, person.job)
        }
    }

    override fun getLoadingView(): RemoteViews? = null

    override fun getViewTypeCount(): Int = 1

    override fun getItemId(position: Int): Long = people[position].name.hashCode().toLong()

    override fun hasStableIds(): Boolean = true

}