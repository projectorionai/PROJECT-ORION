package btools.routingapp;

import android.os.Bundle;

// BRouter's public service contract. See docs/OFFLINE_CAR_ROUTES_2026-09-30.md.
interface IBRouterService {
    String getTrackFromParams(in Bundle params);
}
