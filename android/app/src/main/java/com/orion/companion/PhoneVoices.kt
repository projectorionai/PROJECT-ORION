package com.orion.companion

/** Prebuilt voices shared by Gemini speech generation and the Live API. */
internal object PhoneVoices {
    const val PHONE = "Phone"
    const val DEFAULT = "Charon"
    data class Voice(val name: String, val delivery: String)
    val cloud = listOf(
        Voice("Charon", "Informative"), Voice("Orus", "Firm"), Voice("Algieba", "Smooth"),
        Voice("Schedar", "Even"), Voice("Iapetus", "Clear"), Voice("Algenib", "Gravelly"),
        Voice("Zephyr", "Bright"), Voice("Puck", "Upbeat"), Voice("Kore", "Firm"),
        Voice("Fenrir", "Excitable"), Voice("Leda", "Youthful"), Voice("Aoede", "Breezy"),
        Voice("Callirrhoe", "Relaxed"), Voice("Autonoe", "Bright"), Voice("Enceladus", "Breathy"),
        Voice("Umbriel", "Relaxed"), Voice("Despina", "Smooth"), Voice("Erinome", "Clear"),
        Voice("Rasalgethi", "Informative"), Voice("Laomedeia", "Upbeat"), Voice("Achernar", "Soft"),
        Voice("Alnilam", "Firm"), Voice("Gacrux", "Mature"), Voice("Pulcherrima", "Forward"),
        Voice("Achird", "Friendly"), Voice("Zubenelgenubi", "Casual"), Voice("Vindemiatrix", "Gentle"),
        Voice("Sadachbia", "Lively"), Voice("Sadaltager", "Knowledgeable"), Voice("Sulafat", "Warm"),
    )
    fun valid(value: String): Boolean = value == PHONE || cloud.any { it.name == value }
    fun cloudName(value: String): String = value.takeIf { name -> cloud.any { it.name == name } } ?: DEFAULT
}
