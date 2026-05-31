import { useState } from 'react';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';

export function useImageHandling() {
  const [image, setImage] = useState<string | null>(null);

  const pickImage = async () => {
    let pickerResult = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'] as any, // Using array to avoid deprecation warning
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!pickerResult.canceled) {
      // Resize to 300x300 for the model
      const manipResult = await ImageManipulator.manipulateAsync(
        pickerResult.assets[0].uri,
        [{ resize: { width: 300, height: 300 } }],
        { compress: 1, format: ImageManipulator.SaveFormat.JPEG }
      );
      setImage(manipResult.uri);
      return manipResult.uri;
    }
    return null;
  };

  const clearImage = () => setImage(null);

  return { image, setImage, pickImage, clearImage };
}
