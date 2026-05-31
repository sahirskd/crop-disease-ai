import { useState, useEffect } from 'react';
import { initModel, runInference, InferenceResult } from '../utils/inference';

export function useInference() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modelReady, setModelReady] = useState(false);

  useEffect(() => {
    let isMounted = true;
    async function setup() {
      try {
        await initModel();
        if (isMounted) setModelReady(true);
      } catch (err: any) {
        if (isMounted) setError(err.message || 'Failed to initialize model');
      }
    }
    setup();
    return () => { isMounted = false; };
  }, []);

  const analyze = async (imageUri: string) => {
    if (!imageUri || !modelReady) return;
    setLoading(true);
    setError(null);
    setResult(null);
    
    try {
      const res = await runInference(imageUri);
      setResult(res);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Inference failed.");
    } finally {
      setLoading(false);
    }
  };

  const clearResult = () => {
    setResult(null);
    setError(null);
  };

  return { loading, result, error, modelReady, analyze, clearResult, setError };
}
