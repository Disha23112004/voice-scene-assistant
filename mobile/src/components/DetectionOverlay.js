// src/components/DetectionOverlay.js
// Draws bounding-box labels over the camera view.
// NOTE: boxes from the backend are in 320×320 space; we scale to screen dims.

import React from "react";
import { View, Text, StyleSheet, useWindowDimensions } from "react-native";

const YOLO_SIZE = 320;

const LABEL_COLORS = {
  person:     "#f87171",
  car:        "#60a5fa",
  bicycle:    "#34d399",
  dog:        "#fbbf24",
  cat:        "#a78bfa",
  default:    "#4ade80",
};

function color(label) {
  return LABEL_COLORS[label.toLowerCase()] ?? LABEL_COLORS.default;
}

export default function DetectionOverlay({ objects = [] }) {
  const { width, height } = useWindowDimensions();
  const scaleX = width  / YOLO_SIZE;
  const scaleY = height / YOLO_SIZE;

  if (!objects.length) return null;

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      {objects.map((obj, i) => {
        const [x, y, w, h] = obj.box;
        const left   = x * scaleX;
        const top    = y * scaleY;
        const bw     = w * scaleX;
        const bh     = h * scaleY;
        const c      = color(obj.label);
        return (
          <View
            key={i}
            style={[styles.box, { left, top, width: bw, height: bh, borderColor: c }]}
          >
            <View style={[styles.labelBg, { backgroundColor: c }]}>
              <Text style={styles.labelText}>
                {obj.label} {obj.distance_m ? `${obj.distance_m}m` : ""}
              </Text>
            </View>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  box: {
    position: "absolute",
    borderWidth: 2,
    borderRadius: 4,
  },
  labelBg: {
    position: "absolute",
    top: -20,
    left: -2,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  labelText: {
    color: "#000",
    fontSize: 10,
    fontWeight: "700",
  },
});
