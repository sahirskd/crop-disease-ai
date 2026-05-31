import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, Button, ActivityIndicator } from 'react-native';
import { useImageHandling } from './src/hooks/useImageHandling';
import { useInference } from './src/hooks/useInference';
import { HeatmapOverlay } from './src/components/HeatmapOverlay';
import { ImageSelector } from './src/components/ImageSelector';
import { ResultBanner } from './src/components/ResultBanner';
import { Toast } from './src/components/Toast';

export default function App() {
  const { image, pickImage } = useImageHandling();
  const { loading, result, error, modelReady, analyze, setError } = useInference();

  const handlePickImage = async () => {
    await pickImage();
  };

  const handleRunInference = async () => {
    if (image) {
      await analyze(image);
    }
  };

  return (
    <View style={styles.container}>
      <Toast message={error} onClose={() => setError(null)} />
      
      <Text style={styles.title}>CropSense Edge</Text>
      
      <ImageSelector imageUri={image}>
        {result && result.heatmap && (
          <HeatmapOverlay heatmap={result.heatmap} />
        )}
      </ImageSelector>

      <View style={styles.buttonContainer}>
        <Button title="Pick an Image" onPress={handlePickImage} />
        {image && (
          <Button 
            title={modelReady ? "Analyze" : "Loading Model..."} 
            onPress={handleRunInference} 
            disabled={loading || !modelReady} 
          />
        )}
      </View>

      {loading && <ActivityIndicator size="large" color="#0000ff" />}
      
      <ResultBanner result={result} />
      
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    paddingTop: 80,
    paddingHorizontal: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 30,
  },
  buttonContainer: {
    flexDirection: 'row',
    gap: 20,
    marginTop: 30,
    marginBottom: 20,
  },
});
