// App.js
import React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import HomeScreen   from "./src/screens/HomeScreen";
import MemoryScreen from "./src/screens/MemoryScreen";
import AlertsScreen from "./src/screens/AlertsScreen";

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        screenOptions={{
          headerShown: false,
          animation: "slide_from_right",
          contentStyle: { backgroundColor: "#000" },
        }}
      >
        <Stack.Screen name="Home"    component={HomeScreen}   />
        <Stack.Screen name="Memory"  component={MemoryScreen} />
        <Stack.Screen name="Alerts"  component={AlertsScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
