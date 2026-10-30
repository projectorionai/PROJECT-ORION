# ORION's web UI calls NativeBridge through window.OrionNative. WebView finds
# those methods by reflection, by name, and exposes only the ones that still
# carry @JavascriptInterface at runtime, so R8 must keep both the member names
# and the annotation. Without this the page's `if (window.OrionNative)` check
# passes but every call is silently undefined.
-keepattributes RuntimeVisibleAnnotations
-keepclassmembers class * {
    @android.webkit.JavascriptInterface <methods>;
}

# Everything else the app reaches reflectively (activities, services, the
# accessibility service) is named in the manifest, and AAPT generates keep
# rules for it. OkHttp and AndroidX ship their own consumer rules.
