// src/screens/HomeScreen.js
import React, { useRef, useState, useCallback } from "react";
import {
  View, Text, TouchableOpacity, StyleSheet,
  SafeAreaView, StatusBar, Animated, Vibration, Platform,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { BlurView } from "expo-blur";
import { Ionicons } from "@expo/vector-icons";
import { useVoice } from "../hooks/useVoice";
import DetectionOverlay from "../components/DetectionOverlay";
import StatusStrip from "../components/StatusStrip";
import { analyzeFrame } from "../services/api";

export default function HomeScreen({ navigation }) {
  const cameraRef                       = useRef(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing]             = useState("back");
  const [lastResult, setLastResult]     = useState(null);
  const [autoPulse, setAutoPulse]       = useState(false);
  const pulseAnim                       = useRef(new Animated.Value(1)).current;

  const autoRef = useRef(null);

  const toggleAutoPulse = useCallback(() => {
    setAutoPulse((prev) => {
      if (!prev) {
        autoRef.current = setInterval(async () => {
          if (cameraRef.current) {
            try {
              const photo  = await cameraRef.current.takePictureAsync({ quality: 0.4 });
              const result = await analyzeFrame(photo.uri);
              setLastResult(result);
            } catch (_) {}
          }
        }, 3000);
        Animated.loop(
          Animated.sequence([
            Animated.timing(pulseAnim, { toValue: 1.15, duration: 600, useNativeDriver: true }),
            Animated.timing(pulseAnim, { toValue: 1,    duration: 600, useNativeDriver: true }),
          ])
        ).start();
      } else {
        clearInterval(autoRef.current);
        pulseAnim.stopAnimation();
        pulseAnim.setValue(1);
      }
      return !prev;
    });
  }, [pulseAnim]);

  const handleAnalysis = useCallback((result) => setLastResult(result), []);

  const { isListening, isSpeaking, isProcessing, transcript,
          startListening, stopListening, speak } = useVoice({
    onAnalyze: handleAnalysis,
    cameraRef,
  });

  const snapAndDescribe = useCallback(async () => {
    if (!cameraRef.current) return;
    if (Platform.OS !== "web") Vibration.vibrate(50);
    try {
      const photo  = await cameraRef.current.takePictureAsync({ quality: 0.5 });
      const result = await analyzeFrame(photo.uri);
      setLastResult(result);
      speak(result.description);
    } catch (e) {
      speak("Could not analyze the scene.");
    }
  }, [speak]);

  // ── On web, mic button is a TOGGLE (click to start, click to stop)
  // ── On mobile, mic button is HOLD (press in to start, press out to stop)
  const handleMicPress = useCallback(() => {
    if (Platform.OS === "web") {
      if (isListening) {
        stopListening();
      } else {
        startListening();
      }
    }
  }, [isListening, startListening, stopListening]);

  const getStatusText = () => {
    if (isListening)   return "🎙 Listening… click mic again to stop";
    if (isProcessing)  return "⏳ Processing your command…";
    if (isSpeaking)    return "🔊 " + (lastResult?.description ?? "Speaking…");
    if (transcript)    return `You said: "${transcript}"`;
    if (lastResult?.description) return lastResult.description;
    return Platform.OS === "web"
      ? "Click 👁 to describe or 🎙 to speak a command"
      : "Hold 🎙 to speak or tap 👁 to describe";
  };

  if (!permission) return <View style={styles.container} />;
  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.permissionContainer}>
        <Ionicons name="camera-outline" size={64} color="#fff" />
        <Text style={styles.permissionTitle}>Camera Access Needed</Text>
        <Text style={styles.permissionSub}>
          This app needs camera access to analyze your surroundings.
        </Text>
        <TouchableOpacity style={styles.grantBtn} onPress={requestPermission}>
          <Text style={styles.grantBtnText}>Grant Permission</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />

      <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing={facing}>

        {lastResult?.objects && (
          <DetectionOverlay objects={lastResult.objects} />
        )}

        <StatusStrip
          objects={lastResult?.objects?.length ?? 0}
          alerts={lastResult?.alerts ?? []}
          isProcessing={isProcessing}
        />

        {lastResult?.alerts?.length > 0 && (
          <View style={styles.alertBanner}>
            <Ionicons name="alert-circle" size={16} color="#fff" />
            <Text style={styles.alertText}>{lastResult.alerts[0]}</Text>
          </View>
        )}

        {/* Transcript bubble — shows what you said */}
        {transcript !== "" && !isListening && (
          <View style={styles.transcriptBubble}>
            <Text style={styles.transcriptText}>🗣 "{transcript}"</Text>
          </View>
        )}

        <BlurView intensity={60} tint="dark" style={styles.bottomPanel}>

          <Text style={styles.descriptionText} numberOfLines={3}>
            {getStatusText()}
          </Text>

          <View style={styles.controlsRow}>

            <TouchableOpacity
              style={styles.sideBtn}
              onPress={() => navigation.navigate("Memory")}
            >
              <Ionicons name="time-outline" size={24} color="#fff" />
              <Text style={styles.sideBtnLabel}>Memory</Text>
            </TouchableOpacity>

            {/* Web: tap to toggle | Mobile: hold to talk */}
            <TouchableOpacity
              style={[styles.micBtn, isListening && styles.micBtnActive]}
              onPress={Platform.OS === "web" ? handleMicPress : undefined}
              onPressIn={Platform.OS !== "web" ? startListening : undefined}
              onPressOut={Platform.OS !== "web" ? stopListening : undefined}
              activeOpacity={0.8}
            >
              <Animated.View style={{ transform: [{ scale: isListening ? pulseAnim : 1 }] }}>
                <Ionicons
                  name={isListening ? "mic" : "mic-outline"}
                  size={36}
                  color="#fff"
                />
              </Animated.View>
            </TouchableOpacity>

            <TouchableOpacity style={styles.sideBtn} onPress={snapAndDescribe}>
              <Ionicons name="eye-outline" size={24} color="#fff" />
              <Text style={styles.sideBtnLabel}>Describe</Text>
            </TouchableOpacity>

          </View>

          <View style={styles.secondaryRow}>
            <TouchableOpacity style={styles.chip} onPress={toggleAutoPulse}>
              <Ionicons
                name={autoPulse ? "radio-button-on" : "radio-button-off"}
                size={14}
                color={autoPulse ? "#4ade80" : "#aaa"}
              />
              <Text style={[styles.chipText, autoPulse && { color: "#4ade80" }]}>
                Auto-Scan
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.chip}
              onPress={() => setFacing(f => f === "back" ? "front" : "back")}
            >
              <Ionicons name="camera-reverse-outline" size={14} color="#aaa" />
              <Text style={styles.chipText}>Flip</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.chip}
              onPress={() => navigation.navigate("Alerts")}
            >
              <Ionicons name="notifications-outline" size={14} color="#aaa" />
              <Text style={styles.chipText}>Alerts</Text>
            </TouchableOpacity>
          </View>

        </BlurView>
      </CameraView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container:           { flex: 1, backgroundColor: "#000" },
  permissionContainer: { flex: 1, backgroundColor: "#111", alignItems: "center",
                         justifyContent: "center", padding: 32 },
  permissionTitle:     { color: "#fff", fontSize: 22, fontWeight: "700", marginTop: 16 },
  permissionSub:       { color: "#aaa", fontSize: 15, textAlign: "center", marginTop: 8 },
  grantBtn:            { marginTop: 24, backgroundColor: "#7c3aed", borderRadius: 12,
                         paddingHorizontal: 32, paddingVertical: 14 },
  grantBtnText:        { color: "#fff", fontSize: 16, fontWeight: "600" },

  alertBanner: {
    position: "absolute", top: 60, left: 16, right: 16,
    backgroundColor: "rgba(239,68,68,0.85)", borderRadius: 10,
    flexDirection: "row", alignItems: "center", padding: 10, gap: 8,
  },
  alertText: { color: "#fff", fontSize: 14, fontWeight: "600", flex: 1 },

  transcriptBubble: {
    position: "absolute", top: 100, left: 16, right: 16,
    backgroundColor: "rgba(124,58,237,0.85)", borderRadius: 10,
    padding: 10, alignItems: "center",
  },
  transcriptText: { color: "#fff", fontSize: 13, fontStyle: "italic" },

  bottomPanel: {
    position: "absolute", bottom: 0, left: 0, right: 0,
    paddingBottom: 32, paddingTop: 16, paddingHorizontal: 20,
    borderTopLeftRadius: 24, borderTopRightRadius: 24,
    overflow: "hidden",
  },
  descriptionText: {
    color: "#fff", fontSize: 14, lineHeight: 20, minHeight: 60,
    textAlign: "center", marginBottom: 16,
  },
  controlsRow: {
    flexDirection: "row", alignItems: "center",
    justifyContent: "space-around", marginBottom: 16,
  },
  sideBtn:      { alignItems: "center", gap: 4 },
  sideBtnLabel: { color: "#ccc", fontSize: 11 },
  micBtn: {
    width: 72, height: 72, borderRadius: 36,
    backgroundColor: "#7c3aed",
    alignItems: "center", justifyContent: "center",
    shadowColor: "#7c3aed", shadowOpacity: 0.6,
    shadowRadius: 12, elevation: 8,
  },
  micBtnActive: { backgroundColor: "#ef4444" },

  secondaryRow: { flexDirection: "row", justifyContent: "center", gap: 16 },
  chip: {
    flexDirection: "row", alignItems: "center", gap: 4,
    backgroundColor: "rgba(255,255,255,0.1)",
    paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20,
  },
  chipText: { color: "#aaa", fontSize: 12 },
});