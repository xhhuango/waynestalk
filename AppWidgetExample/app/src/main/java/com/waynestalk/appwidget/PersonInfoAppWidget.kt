package com.waynestalk.appwidget

import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.Context
import android.util.SizeF
import android.widget.RemoteViews

val wayne = Person(
    name = "Wayne",
    job = "Software programmer",
    website = "https://waynestalk.com/",
    github = "https://github.com/xhhuango",
)

/**
 * Implementation of App Widget functionality.
 */
class PersonInfoAppWidget : AppWidgetProvider() {
    override fun onUpdate(
        context: Context,
        appWidgetManager: AppWidgetManager,
        appWidgetIds: IntArray,
    ) {
        // There may be multiple widgets active, so update all of them
        for (appWidgetId in appWidgetIds) {
            updateAppWidget(context, appWidgetManager, appWidgetId)
        }
    }

    override fun onEnabled(context: Context) {
        // Enter relevant functionality for when the first widget is created
    }

    override fun onDisabled(context: Context) {
        // Enter relevant functionality for when the last widget is disabled
    }
}

internal fun updateAppWidget(
    context: Context,
    appWidgetManager: AppWidgetManager,
    appWidgetId: Int,
) {
    // Construct the RemoteViews object
    val simpleView = RemoteViews(context.packageName, R.layout.person_info_app_widget)
    simpleView.setTextViewText(R.id.name_text_view, wayne.name)
    simpleView.setTextViewText(R.id.job_text_view, wayne.job)

    val detailView = RemoteViews(context.packageName, R.layout.person_info_detail_app_widget)
    detailView.setTextViewText(R.id.name_text_view, wayne.name)
    detailView.setTextViewText(R.id.job_text_view, wayne.job)
    detailView.setTextViewText(R.id.website_text_view, wayne.website)
    detailView.setTextViewText(R.id.github_text_view, wayne.github)

    val viewMapping = mapOf(
        SizeF(60f,  100f) to simpleView,
        SizeF(100f, 200f) to detailView,
    )

    val views = RemoteViews(viewMapping)

    // Instruct the widget manager to update the widget
    appWidgetManager.updateAppWidget(appWidgetId, views)
}