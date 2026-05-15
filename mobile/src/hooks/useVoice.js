// src/hooks/useVoice.js
// Always-on continuous voice listening — no button press needed on web.

import { useState, useRef, useCallback, useEffect } from "react";
import { Platform } from "react-native";
import * as Speech from "expo-speech";
import { sendCommand, analyzeFrame, extractText } from "../services/api";

export function useVoice({ onAnalyze, cameraRef }) {
  const [isListening,  setIsListening]  = useState(false);
  const [isSpeaking,   setIsSpeaking]   = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcript,   setTranscript]   = useState("");

  const recognitionRef  = useRef(null);
  const recordingRef    = useRef(null);
  const isProcessingRef = useRef(false);
  const isSpeakingRef   = useRef(false);
  const restartTimerRef = useRef(null);

  // ── Speak text aloud ────────────────────────────────────────────────────────
  const speak = useCallback((text) => {
    if (Platform.OS === "web") {
      window.speechSynthesis.cancel();
      const utt   = new SpeechSynthesisUtterance(text);
      utt.lang    = "en-US";
      utt.rate    = 0.95;
      utt.onstart = () => { setIsSpeaking(true);  isSpeakingRef.current = true;  };
      utt.onend   = () => { setIsSpeaking(false); isSpeakingRef.current = false; };
      utt.onerror = () => { setIsSpeaking(false); isSpeakingRef.current = false; };
      window.speechSynthesis.speak(utt);
    } else {
      setIsSpeaking(true);
      isSpeakingRef.current = true;
      Speech.speak(text, {
        language: "en-US",
        rate: 0.9,
        onDone:  () => { setIsSpeaking(false); isSpeakingRef.current = false; },
        onError: () => { setIsSpeaking(false); isSpeakingRef.current = false; },
      });
    }
  }, []);

  // ── Route the recognised command ────────────────────────────────────────────
  const handleCommand = useCallback(async (command) => {
    if (!command || command.trim().length < 2) return;

    setTranscript(command);
    setIsProcessing(true);
    isProcessingRef.current = true;

    try {
      if (
        command.includes("describe") ||
        command.includes("what do you see") ||
        command.includes("what is") ||
        command.includes("scan") ||
        command.includes("look")
      ) {
        if (cameraRef?.current) {
          const photo  = await cameraRef.current.takePictureAsync({ quality: 0.5 });
          const result = await analyzeFrame(photo.uri);
          onAnalyze?.(result);
          speak(result.description);
          if (result.alerts?.length) {
            setTimeout(() => speak(result.alerts.join(". ")), 2500);
          }
        } else {
          speak("Camera is not ready yet.");
        }
      } else if (
        command.includes("read text") ||
        command.includes("read this") ||
        command.includes("what does it say")
      ) {
        if (cameraRef?.current) {
          const photo  = await cameraRef.current.takePictureAsync({ quality: 0.7 });
          const result = await extractText(photo.uri);
          speak(result.text ? `I can read: ${result.text}` : "I couldn't find any text.");
        }
      } else {
        const result = await sendCommand(command);
        speak(result.response);
      }
    } catch (e) {
      console.error("handleCommand error:", e);
      speak("Something went wrong. Please try again.");
    } finally {
      setIsProcessing(false);
      isProcessingRef.current = false;
    }
  }, [speak, onAnalyze, cameraRef]);

  // ── WEB: Continuous always-on speech recognition ────────────────────────────
  const startContinuousListening = useCallback(() => {
    if (Platform.OS !== "web") return;
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) { console.warn("Web Speech API not supported."); return; }
    if (recognitionRef.current) return; // already running

    const recognition           = new SpeechRecognition();
    recognition.lang            = "en-US";
    recognition.continuous      = false;
    recognition.interimResults  = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => setIsListening(true);

    recognition.onresult = (event) => {
      const command = event.results[0][0].transcript.toLowerCase().trim();
      console.log("Heard:", command);
      recognitionRef.current = null;
      setIsListening(false);
      handleCommand(command);
    };

    recognition.onerror = (event) => {
      recognitionRef.current = null;
      setIsListening(false);
      if (event.error !== "no-speech" && event.error !== "aborted") {
        console.warn("Recognition error:", event.error);
      }
    };

    recognition.onend = () => {
      recognitionRef.current = null;
      setIsListening(false);
      // Auto-restart, but wait if currently speaking or processing
      restartTimerRef.current = setTimeout(() => {
        if (!isProcessingRef.current && !isSpeakingRef.current) {
          startContinuousListening();
        } else {
          const waitInterval = setInterval(() => {
            if (!isProcessingRef.current && !isSpeakingRef.current) {
              clearInterval(waitInterval);
              startContinuousListening();
            }
          }, 500);
        }
      }, 300);
    };

    recognitionRef.current = recognition;
    try { recognition.start(); } catch (e) { recognitionRef.current = null; }
  }, [handleCommand]);

  const stopContinuousListening = useCallback(() => {
    if (restartTimerRef.current) clearTimeout(restartTimerRef.current);
    if (recognitionRef.current) {
      recognitionRef.current.onend = null;
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
    setIsListening(false);
  }, []);

  // ── Auto-start on web when component mounts ─────────────────────────────────
  useEffect(() => {
    if (Platform.OS === "web") {
      const t = setTimeout(() => {
        startContinuousListening();
        speak("Voice assistant ready. Say describe to analyse the scene, or say help for all commands.");
      }, 2000);
      return () => {
        clearTimeout(t);
        stopContinuousListening();
      };
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── MOBILE: hold-to-talk ────────────────────────────────────────────────────
  const startListeningMobile = useCallback(async () => {
    try {
      const { Audio } = await import("expo-av");
      const { status } = await Audio.requestPermissionsAsync();
      if (status !== "granted") { speak("Microphone permission denied."); return; }
      await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });
      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      recordingRef.current = recording;
      setIsListening(true);
    } catch (e) { console.error("startListeningMobile:", e); }
  }, [speak]);

  const stopListeningMobile = useCallback(async () => {
    if (!recordingRef.current) return;
    setIsListening(false);
    isProcessingRef.current = true;
    setIsProcessing(true);
    try {
      await recordingRef.current.stopAndUnloadAsync();
      const uri = recordingRef.current.getURI();
      recordingRef.current = null;
      let command = "";
      const OPENAI_KEY = "";
      if (OPENAI_KEY) {
        try {
          const form = new FormData();
          form.append("file",  { uri, type: "audio/m4a", name: "audio.m4a" });
          form.append("model", "whisper-1");
          const r = await fetch("https://api.openai.com/v1/audio/transcriptions", {
            method: "POST", headers: { Authorization: `Bearer ${OPENAI_KEY}` }, body: form,
          });
          const j = await r.json();
          command = j.text?.toLowerCase() ?? "";
        } catch (_) {}
      }
      await handleCommand(command);
    } catch (e) {
      speak("Something went wrong.");
      setIsProcessing(false);
      isProcessingRef.current = false;
    }
  }, [handleCommand, speak]);

  // ── Public API ───────────────────────────────────────────────────────────────
  return {
    isListening,
    isSpeaking,
    isProcessing,
    transcript,
    startListening: Platform.OS === "web" ? startContinuousListening : startListeningMobile,
    stopListening:  Platform.OS === "web" ? stopContinuousListening  : stopListeningMobile,
    speak,
  };
}