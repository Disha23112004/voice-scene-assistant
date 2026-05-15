// src/screens/AlertsScreen.js
import React, { useState } from "react";
import {
  View, Text, TextInput, TouchableOpacity,
  FlatList, StyleSheet, SafeAreaView,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { sendCommand } from "../services/api";

export default function AlertsScreen({ navigation }) {
  const [alerts,  setAlerts]  = useState([]);
  const [object,  setObject]  = useState("");
  const [type,    setType]    = useState("appear");
  const [feedback, setFeedback] = useState("");

  const addAlert = async () => {
    if (!object.trim()) return;
    const cmd   = type === "appear"
      ? `alert me when ${object.trim()} appears`
      : `alert me if ${object.trim()} disappears`;
    const res   = await sendCommand(cmd);
    setAlerts(prev => [...prev, { object: object.trim(), type }]);
    setFeedback(res.response);
    setObject("");
  };

  const removeAlert = async (item) => {
    await sendCommand(`remove alert for ${item.object}`);
    setAlerts(prev => prev.filter(a => a !== item));
  };

  const clearAll = async () => {
    await sendCommand("clear alerts");
    setAlerts([]);
    setFeedback("All alerts cleared.");
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Ionicons name="arrow-back" size={24} color="#fff" />
        </TouchableOpacity>
        <Text style={styles.title}>Object Alerts</Text>
        <TouchableOpacity onPress={clearAll}>
          <Text style={styles.clearText}>Clear all</Text>
        </TouchableOpacity>
      </View>

      {/* Add alert form */}
      <View style={styles.form}>
        <TextInput
          style={styles.input}
          placeholder="Object name (e.g. person, phone)"
          placeholderTextColor="#555"
          value={object}
          onChangeText={setObject}
        />
        <View style={styles.typeRow}>
          {["appear", "disappear"].map(t => (
            <TouchableOpacity
              key={t}
              style={[styles.typeBtn, type === t && styles.typeBtnActive]}
              onPress={() => setType(t)}
            >
              <Text style={[styles.typeBtnText, type === t && styles.typeBtnTextActive]}>
                {t === "appear" ? "🟢 Appears" : "🔴 Disappears"}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
        <TouchableOpacity style={styles.addBtn} onPress={addAlert}>
          <Ionicons name="add" size={20} color="#fff" />
          <Text style={styles.addBtnText}>Set Alert</Text>
        </TouchableOpacity>
        {!!feedback && <Text style={styles.feedback}>{feedback}</Text>}
      </View>

      {/* Alert list */}
      <FlatList
        data={alerts}
        keyExtractor={(_, i) => String(i)}
        contentContainerStyle={{ padding: 16, gap: 10 }}
        ListEmptyComponent={
          <View style={styles.empty}>
            <Ionicons name="notifications-off-outline" size={40} color="#333" />
            <Text style={styles.emptyText}>No alerts set yet.</Text>
          </View>
        }
        renderItem={({ item }) => (
          <View style={styles.alertCard}>
            <Ionicons
              name={item.type === "appear" ? "eye" : "eye-off"}
              size={20}
              color={item.type === "appear" ? "#4ade80" : "#f87171"}
            />
            <View style={styles.alertInfo}>
              <Text style={styles.alertObject}>{item.object}</Text>
              <Text style={styles.alertType}>
                Notify when {item.type === "appear" ? "appears" : "disappears"}
              </Text>
            </View>
            <TouchableOpacity onPress={() => removeAlert(item)}>
              <Ionicons name="close-circle-outline" size={22} color="#666" />
            </TouchableOpacity>
          </View>
        )}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container:  { flex: 1, backgroundColor: "#0f0f0f" },
  header:     {
    flexDirection: "row", alignItems: "center", justifyContent: "space-between",
    padding: 16, borderBottomWidth: 1, borderBottomColor: "#1f1f1f",
  },
  title:      { color: "#fff", fontSize: 18, fontWeight: "700" },
  clearText:  { color: "#ef4444", fontSize: 14 },
  form:       { padding: 16, gap: 10 },
  input:      {
    backgroundColor: "#1a1a1a", borderRadius: 12, color: "#fff",
    padding: 14, fontSize: 15, borderWidth: 1, borderColor: "#2a2a2a",
  },
  typeRow:    { flexDirection: "row", gap: 10 },
  typeBtn:    {
    flex: 1, padding: 10, borderRadius: 10, backgroundColor: "#1a1a1a",
    alignItems: "center", borderWidth: 1, borderColor: "#2a2a2a",
  },
  typeBtnActive:     { borderColor: "#7c3aed", backgroundColor: "#2d1b69" },
  typeBtnText:       { color: "#666", fontSize: 14 },
  typeBtnTextActive: { color: "#fff", fontWeight: "600" },
  addBtn: {
    flexDirection: "row", alignItems: "center", justifyContent: "center",
    gap: 8, backgroundColor: "#7c3aed", borderRadius: 12, padding: 14,
  },
  addBtnText: { color: "#fff", fontSize: 16, fontWeight: "600" },
  feedback:   { color: "#4ade80", fontSize: 13, textAlign: "center" },
  empty:      { alignItems: "center", marginTop: 40, gap: 8 },
  emptyText:  { color: "#444", fontSize: 15 },
  alertCard:  {
    backgroundColor: "#1a1a1a", borderRadius: 14, padding: 14,
    flexDirection: "row", alignItems: "center", gap: 12,
  },
  alertInfo:   { flex: 1 },
  alertObject: { color: "#fff", fontSize: 15, fontWeight: "600", textTransform: "capitalize" },
  alertType:   { color: "#666", fontSize: 13, marginTop: 2 },
});
