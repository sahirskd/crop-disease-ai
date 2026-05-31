import React from 'react';
import { View, StyleSheet } from 'react-native';

interface HeatmapOverlayProps {
  heatmap: number[][];
}

export function HeatmapOverlay({ heatmap }: HeatmapOverlayProps) {
  const H = 10;
  const W = 10;
  const blocks = [];
  
  for (let h = 0; h < H; h++) {
    for (let w = 0; w < W; w++) {
      const val = heatmap[h][w];
      const red = Math.floor(255 * val);
      const blue = Math.floor(255 * (1 - val));
      const alpha = 0.2 + (val * 0.5); 
      
      blocks.push(
        <View
          key={`${h}-${w}`}
          style={{
            position: 'absolute',
            top: `${h * 10}%`,
            left: `${w * 10}%`,
            width: '10%',
            height: '10%',
            backgroundColor: `rgba(${red}, 0, ${blue}, ${alpha})`
          }}
        />
      );
    }
  }
  return <View style={StyleSheet.absoluteFill}>{blocks}</View>;
}
