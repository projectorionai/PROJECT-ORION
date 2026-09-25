package com.orion.companion

import android.content.Intent
import android.os.Bundle
import android.view.inputmethod.EditorInfo
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
        binding.portInput.setText(prefs.getInt(Prefs.PORT, Prefs.DEFAULT_PORT).toString())
        binding.httpsSwitch.isChecked = prefs.getBoolean(Prefs.HTTPS, true)

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
            // A port in the pasted address wins over the Port box.
            val portText = binding.portInput.text.toString().trim()
            val port = address.port
                ?: if (portText.isEmpty()) Prefs.DEFAULT_PORT
                else portText.toIntOrNull()?.takeIf { it in 1..65535 }
            if (port == null) {
                binding.portInput.error = getString(R.string.invalid_port)
                binding.portInput.requestFocus()
                return@setOnClickListener
            }
            // So does an explicit http:// or https:// in it.
            address.https?.let { binding.httpsSwitch.isChecked = it }
            binding.hostInput.setText(address.host)
            binding.portInput.setText(port.toString())

            prefs.edit()
                .putString(Prefs.HOST, address.host)
                .putInt(Prefs.PORT, port)
                .putBoolean(Prefs.HTTPS, binding.httpsSwitch.isChecked)
                .apply()
            // A manual entry is a deliberate override — drop any endpoints learned
            // from a previous pairing so this host is used until we re-pair.
            Prefs.clearEndpoints(this)
            // A new server may present a new certificate; pin it afresh.
            Prefs.clearCertPin(this)
            startActivity(
                Intent(this, MainActivity::class.java)
                    .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                    .putExtra(MainActivity.EXTRA_RECONFIGURED, true)
            )
            finish()
        }
    }
}
