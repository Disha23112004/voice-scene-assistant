// src/screens/MemoryScreen.js
import React, { useEffect, useState, useCallback } from "react";
import {
  View, Text, FlatList, TouchableOpacity,
  StyleSheet, SafeAreaView, ActivityIndicator, Alert,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { getMemory, clearMemory } from "../services/api";

function timeAgo(isoString) {
  if (!isoString) return "never";
  const secs = (Date.now() - new Date(isoString).getTime()) / 1000;
  if (secs < 60)   return `${Math.round(secs)}s ago`;
  if (secs < 3600) return `${Math.round(secs / 60)}m ago`;
  return `${Math.round(secs / 3600)}h ago`;
}

export default function MemoryScreen({ navigation }) {
  const [memory,  setMemory]  = useState({});
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getMemory();
      setMemory(data.memory ?? {});
    } catch (e) {
      Alert.alert("Error", "Could not load memory from server.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleClear = () =>
    Alert.alert("Clear Memory", "Delete all scene history?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Clear",
        style: "destructive",
        onPress: async () => {
          await clearMemory();
          setMemory({});
        },
      },
    ]);

  const sorted = Object.entries(memory).sort(([, a], [, b]) => b.count - a.count);

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Ionicons name="arrow-back" size={24} color="#fff" />
        </TouchableOpacity>
        <Text style={styles.title}>Scene Memory</Text>
        <TouchableOpacity onPress={handleClear}>
          <Ionicons name="trash-outline" size={22} color="#ef4444" />
        </TouchableOpacity>
      </View>

      {loading ? (
        <ActivityIndicator color="#7c3aed" style={{ marginTop: 40 }} />
      ) : sorted.length === 0 ? (
        <View style={styles.empty}>
          <Ionicons name="eye-off-outline" size={48} color="#444" />
          <Text style={styles.emptyText}>Nothing recorded yet.</Text>
          <Text style={styles.emptySub}>
            Point the camera at objects and tap Describe.
          </Text>
        </View>
      ) : (
        <FlatList
          data={sorted}
          keyExtractor={([label]) => label}
          contentContainerStyle={{ padding: 16, gap: 10 }}
          renderItem={({ item: [label, info] }) => (
            <View style={styles.card}>
              <View style={styles.cardLeft}>
                <Text style={styles.cardLabel}>{label}</Text>
                <Text style={styles.cardSub}>
                  Last seen {timeAgo(info.last_seen)}
                </Text>
              </View>
              <View style={styles.badge}>
                <Text style={styles.badgeText}>{info.count}×</Text>
              </View>
            </View>
          )}
          onRefresh={load}
          refreshing={loading}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0f0f0f" },
  header: {
    flexDirection: "row", alignItems: "center", justifyContent: "space-between",
    padding: 16, borderBottomWidth: 1, borderBottomColor: "#1f1f1f",
  },
  title:    { color: "#fff", fontSize: 18, fontWeight: "700" },
  empty:    { flex: 1, alignItems: "center", justifyContent: "center", gap: 8 },
  emptyText:{ color: "#555", fontSize: 18, fontWeight: "600" },
  emptySub: { color: "#444", fontSize: 13, textAlign: "center", paddingHorizontal: 32 },
  card: {
    backgroundColor: "#1a1a1a", borderRadius: 14,
    padding: 16, flexDirection: "row",
    alignItems: "center", justifyContent: "space-between",
  },
  cardLeft:  { flex: 1, gap: 4 },
  cardLabel: { color: "#fff", fontSize: 16, fontWeight: "600", textTransform: "capitalize" },
  cardSub:   { color: "#666", fontSize: 13 },
  badge: {
    backgroundColor: "#7c3aed", borderRadius: 20,
    paddingHorizontal: 10, paddingVertical: 4,
  },
  badgeText: { color: "#fff", fontSize: 13, fontWeight: "700" },
});
