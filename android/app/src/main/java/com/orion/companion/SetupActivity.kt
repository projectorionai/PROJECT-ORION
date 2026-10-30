package com.orion.companion

import android.app.AlertDialog
import android.content.Intent
import android.os.Bundle
import android.text.InputType
import android.view.inputmethod.EditorInfo
import android.widget.EditText
import androidx.appcompat.app.AppCompatActivity
import com.orion.companion.databinding.ActivitySetupBinding

/** First-run (and "change server") screen: where is ORION on the network? */
class SetupActivity : AppCompatActivity() {

    private lateinit var binding: ActivitySetupBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySetupBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val prefs = getSharedPreferences(Prefs.NAME, MODE_PRIVATE)
        binding.hostInput.setText(prefs.getString(Prefs.HOST, ""))
        val oldPort = prefs.getInt(Prefs.PORT, Prefs.DEFAULT_PORT)
        // Legacy installations used HTTP on these ports. Both supported HTTPS
        // tailnet routes now use 443, including the route without Serve.
        binding.portInput.setText((if (oldPort in setOf(8765, 8000)) 443 else oldPort).toString())
        binding.httpsSwitch.isChecked = true
        binding.httpsSwitch.isEnabled = false

        binding.portInput.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_DONE) {
                binding.connectButton.performClick()
                true
            } else {
                false
            }
        }

        binding.connectButton.setOnClickListener {
            val raw = binding.hostInput.text.toString()
            if (raw.isBlank()) {
                binding.hostInput.error = getString(R.string.need_host)
                binding.hostInput.requestFocus()
                return@setOnClickListener
            }
            val address = Prefs.parseAddress(raw)
            if (address == null) {
                binding.hostInput.error = getString(R.string.invalid_host)
                binding.hostInput.requestFocus()
                return@setOnClickListener
            }
            if (address.https == false) {
                binding.hostInput.error = getString(R.string.https_required)
                binding.hostInput.requestFocus()
                return@setOnClickListener
            }
            if (address.host.matches(Regex("100\\.\\d+\\.\\d+\\.\\d+"))) {
                binding.hostInput.error = getString(R.string.magicdns_required)
                binding.hostInput.requestFocus()
                return@setOnClickListener
            }
            // A port in the pasted address wins over the Port box.
            val portText = binding.portInput.text.toString().trim()
            val port = address.port
                // A home Wi-Fi address means ORION's LAN link, port 8765.
                ?: if (LanTrust.isPrivateLanHost(address.host) && portText in setOf("", "443")) 8765
                else if (portText.isEmpty()) Prefs.DEFAULT_PORT
                else portText.toIntOrNull()?.takeIf { it in 1..65535 }
            if (port == null) {
                binding.portInput.error = getString(R.string.invalid_port)
                binding.portInput.requestFocus()
                return@setOnClickListener
            }
            // So does an explicit http:// or https:// in it.
            binding.httpsSwitch.isChecked = true
            binding.hostInput.setText(address.host)
            binding.portInput.setText(port.toString())

            if (LanTrust.isPrivateLanHost(address.host) && LanTrust.pinFor(this, address.host, port) == null)
                askCertificateCode(address.host, port)
            else finishSetup(address.host, port, null)
        }
    }

    /** A home Wi-Fi address: ORION's certificate is his own, so the owner
     *  types the certificate code the PC shows and the phone checks it
     *  against what this address presents before anything else is sent. */
    private fun askCertificateCode(host: String, port: Int) {
        val input = EditText(this).apply {
            hint = "1A2B-3C4D-5E6F-7A8B"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_CHARACTERS
        }
        AlertDialog.Builder(this).setTitle("PC certificate code")
            .setMessage("On the PC open SECURITY → PAIR PHONE and type the certificate code shown for this Wi-Fi address.")
            .setView(input)
            .setPositiveButton("Check") { _, _ ->
                val typed = input.text.toString()
                binding.connectButton.isEnabled = false
                Thread {
                    val presented = runCatching { LanTrust.presentedFingerprint(host, port) }.getOrNull()
                    runOnUiThread {
                        if (isFinishing || isDestroyed) return@runOnUiThread
                        binding.connectButton.isEnabled = true
                        val problem = when {
                            presented == null ->
                                "The PC did not answer at $host:$port. Check it is on and on this Wi-Fi."
                            !LanTrust.codeMatches(presented, typed) ->
                                "That certificate code does not match this address. Nothing was sent."
                            else -> null
                        }
                        if (problem != null) binding.hostInput.error = problem
                        else if (presented != null) finishSetup(host, port, presented)
                    }
                }.apply { isDaemon = true; name = "orion-lan-check" }.start()
            }
            .setNegativeButton("Cancel", null).show()
    }

    private fun finishSetup(host: String, port: Int, verifiedPin: String?) {
        val prefs = getSharedPreferences(Prefs.NAME, MODE_PRIVATE)
        val serverChanged = !prefs.getString(Prefs.HOST, "").orEmpty()
            .equals(host, ignoreCase = true) ||
            prefs.getInt(Prefs.PORT, Prefs.DEFAULT_PORT) != port ||
            prefs.getBoolean(Prefs.HTTPS, true) != binding.httpsSwitch.isChecked

        prefs.edit()
            .putString(Prefs.HOST, host)
            .putInt(Prefs.PORT, port)
            .putBoolean(Prefs.HTTPS, binding.httpsSwitch.isChecked)
            .apply()
        // The pin this exact address already earned (the code prompt was
        // skipped because of it) survives the clean-up below: wiping it left
        // the phone unable to trust the PC on Wi-Fi again (review, 29 Oct 2026).
        val pin = verifiedPin ?: LanTrust.pinFor(this, host, port)
        if (serverChanged) {
            // Never seed a previous server's refresh token into a new
            // origin. A changed endpoint must pair afresh.
            Prefs.clearEndpoints(this)
            Prefs.clearCertPin(this)
            Prefs.clearCredentials(this)
            LanTrust.clearPins(this)
        }
        if (pin != null) LanTrust.savePin(this, host, port, pin)
        startActivity(
            Intent(this, MainActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                .putExtra(MainActivity.EXTRA_RECONFIGURED, true)
        )
        finish()
    }
}
