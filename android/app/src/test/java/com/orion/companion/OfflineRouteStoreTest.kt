package com.orion.companion

import java.io.File
import java.io.IOException
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class OfflineRouteStoreTest {
    @get:Rule val temporary = TemporaryFolder()

    private fun track(name: String) = """
        <gpx><trk><name>$name</name><trkseg>
          <trkpt lat="52.0" lon="-1.0"/><trkpt lat="52.001" lon="-1.0"/>
        </trkseg></trk></gpx>
    """.trimIndent()

    @Test fun routeRestoresAfterStoreRecreationWithoutNetworkOrAnEngine() {
        val file = File(temporary.root, "private/route.gpx")
        val store = OfflineRouteStore(file)
        assertNull(store.read())
        val written = store.save(track("Journey"), store.beginChange())
        val restored = OfflineRouteStore(file).read()!!
        assertEquals(written.name, restored.name)
        assertEquals(written.points, restored.points)
        assertEquals(written.distanceMetres, restored.distanceMetres, .001)
        assertFalse(File(file.parentFile, "route.gpx.pending").exists())
    }

    @Test fun lateOlderCalculationCannotReplaceANewerScreensRoute() {
        val file = File(temporary.root, "route.gpx")
        val older = OfflineRouteStore(file)
        val newer = OfflineRouteStore(file)
        val oldRevision = older.beginChange()
        newer.save(track("New journey"), newer.beginChange())
        assertThrows(IOException::class.java) { older.save(track("Old journey"), oldRevision) }
        assertEquals("New journey", newer.read()!!.name)
    }

    @Test fun cancelledCalculationAndMalformedImportPreserveTheLastUsableRoute() {
        val file = File(temporary.root, "route.gpx")
        val store = OfflineRouteStore(file)
        store.save(track("Usable"), store.beginChange())
        val original = file.readBytes()
        assertThrows(IOException::class.java) {
            store.save(track("Cancelled"), store.beginChange()) { false }
        }
        assertThrows(IllegalArgumentException::class.java) {
            store.save("<gpx/>", store.beginChange())
        }
        assertArrayEquals(original, file.readBytes())
        assertEquals("Usable", store.read()!!.name)
    }

    @Test fun staleQueuedClearDoesNotEraseANewerJourney() {
        val file = File(temporary.root, "route.gpx")
        val older = OfflineRouteStore(file)
        val newer = OfflineRouteStore(file)
        val clearRevision = older.beginChange()
        newer.save(track("Keep"), newer.beginChange())
        older.clear(clearRevision)
        assertEquals("Keep", newer.read()!!.name)
        newer.clear(newer.beginChange())
        assertNull(newer.read())
    }

    @Test fun unreadableOversizedAndInvalidUtf8StoresAreRejectedWithoutDeletion() {
        val file = temporary.newFile("route.gpx")
        val store = OfflineRouteStore(file)
        file.writeBytes(byteArrayOf(0xc3.toByte(), 0x28))
        assertThrows(Exception::class.java) { store.read() }
        assertTrue(file.exists())
        java.io.RandomAccessFile(file, "rw").use { it.setLength(OfflineRouteGpx.MAX_BYTES + 1L) }
        assertThrows(IOException::class.java) { store.read() }
        assertEquals(OfflineRouteGpx.MAX_BYTES + 1L, file.length())
    }

    @Test fun rerouteReservationCannotAdoptAnotherScreensJourney() {
        val file = File(temporary.root, "route.gpx")
        val navigation = OfflineRouteStore(file)
        val map = OfflineRouteStore(file)
        val originalRevision = navigation.beginChange()
        navigation.save(track("Original"), originalRevision)
        val snapshot = navigation.readSnapshot()
        assertEquals("Original", snapshot.first!!.name)
        assertEquals(originalRevision, snapshot.second)
        val rerouteRevision = navigation.beginChangeIfCurrent(originalRevision)!!
        assertFalse(navigation.isCurrent(originalRevision))
        val newerRevision = map.beginChange()
        map.save(track("Different journey"), newerRevision)
        assertNull(navigation.beginChangeIfCurrent(rerouteRevision))
        assertThrows(IOException::class.java) { navigation.save(track("Old reroute"), rerouteRevision) }
        assertEquals("Different journey", navigation.readSnapshot().first!!.name)
        assertEquals(newerRevision, navigation.currentRevision())
    }
}
