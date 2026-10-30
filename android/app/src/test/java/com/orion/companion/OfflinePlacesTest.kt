package com.orion.companion

import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.DataOutputStream

class OfflinePlacesTest {
    @get:Rule val temporary = TemporaryFolder()
    private fun fails(action: () -> Unit) {
        try { action(); fail("Expected saved place rejection") } catch (_: Exception) { }
    }

    @Test fun namedCoordinatesSurviveStoreRecreationAndCanBeRemoved() {
        val file = java.io.File(temporary.root, "private/places.bin")
        val store = OfflinePlaces(file)
        assertTrue(store.list().isEmpty())
        val home = store.save(" Home ", OfflineCoordinate(52.0, -1.0))
        val work = store.save("Work", OfflineCoordinate(52.01, -1.02))
        assertEquals(listOf(home, work), OfflinePlaces(file).list())
        assertEquals("Home", home.name)
        store.remove(home.id)
        assertEquals(listOf(work), OfflinePlaces(file).list())
        assertFalse(java.io.File(file.parentFile, "places.bin.pending").exists())
    }

    @Test fun corruptStoreIsNotSilentlyOverwrittenByANewPlace() {
        val file = temporary.newFile("places.bin").apply { writeText("corrupt data") }
        val original = file.readBytes()
        fails { OfflinePlaces(file).save("Home", OfflineCoordinate(52.0, -1.0)) }
        assertArrayEquals(original, file.readBytes())
    }

    @Test fun impossibleCountsAndTrailingDataAreRejected() {
        val file = temporary.newFile("places.bin")
        DataOutputStream(file.outputStream()).use { it.writeInt(0x4f525031); it.writeInt(129) }
        fails { OfflinePlaces(file).list() }
        DataOutputStream(file.outputStream()).use { it.writeInt(0x4f525031); it.writeInt(0); it.writeByte(1) }
        fails { OfflinePlaces(file).list() }
    }

    @Test fun namesAndCapacityAreBounded() {
        val file = temporary.newFile("unused").also { it.delete() }
        val store = OfflinePlaces(file)
        listOf("", " ", "x".repeat(81), "a\nb").forEach { name ->
            fails { store.save(name, OfflineCoordinate(52.0, -1.0)) }
        }
        repeat(128) { store.save("Place $it", OfflineCoordinate(52.0, -1.0)) }
        fails { store.save("Extra", OfflineCoordinate(52.0, -1.0)) }
        assertEquals(128, store.list().size)
    }

    @Test fun removingUnknownPlaceDoesNotChangeTheStore() {
        val file = java.io.File(temporary.root, "places.bin")
        val store = OfflinePlaces(file)
        val place = store.save("Home", OfflineCoordinate(52.0, -1.0))
        fails { store.remove("unknown") }
        assertEquals(listOf(place), store.list())
    }
}
