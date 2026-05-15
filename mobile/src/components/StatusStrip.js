// src/components/StatusStrip.js
import React from "react";
import { View, Text, StyleSheet, ActivityIndicator } from "react-native";

export default function StatusStrip({ objects, alerts, isProcessing }) {
  return (
    <View style={styles.strip}>
      <Pill label={`${objects} objects`} color="#4ade80" />
      {alerts.length > 0 && <Pill label={`${alerts.length} alert`} color="#f87171" />}
      {isProcessing && <ActivityIndicator size="small" color="#fff" style={{ marginLeft: 6 }} />}
    </View>
  );
}

function Pill({ label, color }) {
  return (
    <View style={[styles.pill, { borderColor: color }]}>
      <View style={[styles.dot, { backgroundColor: color }]} />
      <Text style={styles.pillText}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  strip: {
    position: "absolute",
    top: 12, left: 12,
    flexDirection: "row", gap: 8,
  },
  pill: {
    flexDirection: "row", alignItems: "center", gap: 5,
    backgroundColor: "rgba(0,0,0,0.55)",
    paddingHorizontal: 10, paddingVertical: 5,
    borderRadius: 20, borderWidth: 1,
  },
  dot:      { width: 7, height: 7, borderRadius: 4 },
  pillText: { color: "#fff", fontSize: 11, fontWeight: "600" },
});
