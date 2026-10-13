import {
  ModelId,
  ModelName,
  PredictionClass,
  PredictionResult,
  RiskLevel,
  SingleModelPrediction,
  URLFeatures,
} from '../types';
import { extractURLFeatures } from './featureExtractor';

export const ORDERED_FEATURE_KEYS: (keyof URLFeatures)[] = [
  'url_length',
  'valid_url',
  'at_symbol',
  'sensitive_words_count',
  'path_length',
  'isHttps',
  'nb_dots',
  'nb_hyphens',
  'nb_and',
  'nb_or',
  'nb_www',
  'nb_com',
  'nb_underscore',
];

// StandardScaler parameters (mean & standard deviation) fitted on the 1,354 unique URL dataset
export const SCALER_PARAMS: Record<keyof URLFeatures, { mean: number; scale: number }> = {
  url_length: { mean: 51.7947, scale: 20.0934 },
  valid_url: { mean: 0.1196, scale: 0.3247 },
  at_symbol: { mean: 0.0155, scale: 0.3315 },
  sensitive_words_count: { mean: 0.1507, scale: 0.4080 },
  path_length: { mean: 15.3685, scale: 19.3209 },
  isHttps: { mean: 0.2777, scale: 0.4480 },
  nb_dots: { mean: 3.9535, scale: 2.1650 },
  nb_hyphens: { mean: 0.7755, scale: 1.4021 },
  nb_and: { mean: 0.0251, scale: 0.1612 },
  nb_or: { mean: 0.2622, scale: 0.5056 },
  nb_www: { mean: 0.2962, scale: 0.4773 },
  nb_com: { mean: 0.6758, scale: 0.5642 },
  nb_underscore: { mean: 0.1544, scale: 0.5759 },
};

// Trained model weights extracted directly from scikit-learn models (Cell 31 of Notebook)
export const MODEL_WEIGHTS: Record<
  'standard' | 'ridge' | 'lasso',
  {
    name: ModelName;
    intercept: number;
    weights: Record<keyof URLFeatures, number>;
    regularization: string;
    penalty: string;
  }
> = {
  standard: {
    name: 'Standard Logistic Regression',
    intercept: -0.701,
    regularization: 'None (Unconstrained)',
    penalty: 'Standard Log-Loss (C=1.0, max_iter=1000)',
    weights: {
      url_length: 0.609885,
      valid_url: 1.169995,
      at_symbol: -0.529703,
      sensitive_words_count: 0.169712,
      path_length: -0.532479,
      isHttps: 0.085866,
      nb_dots: -0.072254,
      nb_hyphens: 0.204412,
      nb_and: -0.072373,
      nb_or: -0.037641,
      nb_www: -0.535779,
      nb_com: 0.119909,
      nb_underscore: -0.196200,
    },
  },
  ridge: {
    name: 'Ridge Logistic Regression (L2)',
    intercept: -0.701,
    regularization: 'L2 (Ridge)',
    penalty: 'C=1.0, solver="liblinear" (Weight magnitude shrinkage)',
    weights: {
      url_length: 0.608466,
      valid_url: 1.172258,
      at_symbol: -0.519325,
      sensitive_words_count: 0.169749,
      path_length: -0.531957,
      isHttps: 0.085753,
      nb_dots: -0.071586,
      nb_hyphens: 0.204337,
      nb_and: -0.072170,
      nb_or: -0.037697,
      nb_www: -0.534345,
      nb_com: 0.119505,
      nb_underscore: -0.195375,
    },
  },
  lasso: {
    name: 'Lasso Logistic Regression (L1)',
    intercept: -0.701,
    regularization: 'L1 (Lasso)',
    penalty: 'C=1.0, solver="liblinear" (L1 absolute regularization penalty)',
    weights: {
      url_length: 0.565555,
      valid_url: 1.175929,
      at_symbol: -0.347045,
      sensitive_words_count: 0.161626,
      path_length: -0.510538,
      isHttps: 0.078034,
      nb_dots: -0.038321,
      nb_hyphens: 0.205401,
      nb_and: -0.066288,
      nb_or: -0.029815,
      nb_www: -0.533148,
      nb_com: 0.104720,
      nb_underscore: -0.186068,
    },
  },
};

function sigmoid(z: number): number {
  if (z < -40) return 0;
  if (z > 40) return 1;
  return 1 / (1 + Math.exp(-z));
}

export const KNOWN_LEGITIMATE_DOMAINS = [
  'google.com',
  'google.co.in',
  'youtube.com',
  'github.com',
  'microsoft.com',
  'apple.com',
  'amazon.com',
  'amazon.in',
  'wikipedia.org',
  'wikimedia.org',
  'stackoverflow.com',
  'stackexchange.com',
  'run.app',
  'mozilla.org',
  'twitter.com',
  'x.com',
  'linkedin.com',
  'netflix.com',
  'facebook.com',
  'instagram.com',
  'reddit.com',
  'yahoo.com',
  'cloudflare.com',
  'medium.com',
  'pypi.org',
  'npmjs.com',
  'stanford.edu',
  'mit.edu',
  'harvard.edu',
  'berkeley.edu',
  'openai.com',
  'anthropic.com',
  'huggingface.co',
  'arxiv.org',
  'oracle.com',
  'ibm.com',
  'salesforce.com',
  'adobe.com',
  'gitlab.com',
  'atlassian.com',
  'bitbucket.org',
  'dropbox.com',
  'zoom.us',
  'quora.com',
  'nytimes.com',
  'cnn.com',
  'bbc.com',
  'reuters.com',
  'scikit-learn.org',
  'python.org',
  'react.dev',
  'vite.dev',
  'vitejs.dev',
  'tailwindcss.com',
  'nodejs.org',
  'typescriptlang.org',
  'w3schools.com',
  'geeksforgeeks.org',
  'kaggle.com',
];

export interface DomainAnalysis {
  hostname: string;
  isKnownAuthority: boolean;
  hasAtSymbolUserinfo: boolean;
  isRawIp: boolean;
  domainHyphens: number;
}

export function parseDomainInfo(rawUrl: string): DomainAnalysis {
  let normalized = (rawUrl || '').trim();
  if (!/^https?:\/\//i.test(normalized)) {
    normalized = 'http://' + normalized;
  }
  let hostname = '';
  try {
    const parsed = new URL(normalized);
    hostname = parsed.hostname.toLowerCase();
  } catch {
    hostname = '';
  }

  const isRawIp = /^(\d{1,3}\.){3}\d{1,3}$/.test(hostname) || hostname.startsWith('[');
  const isKnownAuthority =
    !isRawIp &&
    KNOWN_LEGITIMATE_DOMAINS.some((d) => hostname === d || hostname.endsWith('.' + d));
  const hasAtSymbolUserinfo = rawUrl.includes('@');
  const domainHyphens = (hostname.match(/-/g) || []).length;

  return {
    hostname,
    isKnownAuthority,
    hasAtSymbolUserinfo,
    isRawIp,
    domainHyphens,
  };
}

export function runModelInference(
  features: URLFeatures,
  modelId: 'standard' | 'ridge' | 'lasso',
  decisionThreshold: number = 0.5,
  calibration?: {
    isHarmonizedLegitimate?: boolean;
    isConfirmedPhishing?: boolean;
  }
): SingleModelPrediction {
  const modelConfig = MODEL_WEIGHTS[modelId];

  // 1. StandardScaler normal transformation: (x - mean) / scale
  let logit = modelConfig.intercept;
  for (const key of ORDERED_FEATURE_KEYS) {
    const rawVal = features[key];
    const { mean, scale } = SCALER_PARAMS[key];
    const scaledVal = (rawVal - mean) / (scale || 1);
    const weight = modelConfig.weights[key];
    logit += weight * scaledVal;
  }

  // 2. Raw Sigmoid probability from trained weights
  const rawProbability = sigmoid(logit);

  // 3. Calibrate probability to eliminate false-positives on benign URLs with 0 threats
  let calibratedProbability = rawProbability;
  if (calibration?.isHarmonizedLegitimate) {
    // URL has zero structural threats and strong benign hallmarks:
    // Bound calibrated probability below 0.22 so verdict agrees with benign indicators
    calibratedProbability = Math.min(rawProbability * 0.32, 0.20);
  } else if (calibration?.isConfirmedPhishing) {
    // URL exhibits undeniable phishing hallmarks (raw IP, @ userinfo attack, spoofed subdomains)
    calibratedProbability = Math.max(rawProbability, 0.78);
  }

  const isPhishing = calibratedProbability >= decisionThreshold;

  const confidence = isPhishing
    ? Math.round(calibratedProbability * 1000) / 10
    : Math.round((1 - calibratedProbability) * 1000) / 10;

  return {
    modelId,
    modelName: modelConfig.name,
    prediction: isPhishing ? 'phishing' : 'legitimate',
    probability: Math.round(calibratedProbability * 1000) / 1000,
    confidence,
    regularization: modelConfig.regularization,
    l1_l2_penalty: modelConfig.penalty,
  };
}

export function evaluateRisk(
  features: URLFeatures,
  phishProb: number,
  rawUrl: string = ''
): {
  riskLevel: RiskLevel;
  riskFactors: string[];
  safeFactors: string[];
  isHarmonizedLegitimate: boolean;
  isConfirmedPhishing: boolean;
} {
  const domainInfo = parseDomainInfo(rawUrl);
  const riskFactors: string[] = [];
  const safeFactors: string[] = [];

  // 1. Host syntax / IP Anomaly
  if (domainInfo.isRawIp || features.valid_url === 1) {
    riskFactors.push('Raw numeric IP address host or non-standard port syntax anomaly');
  } else {
    safeFactors.push('Standard registered domain format (no raw IP or port anomalies)');
  }

  // 2. Known domain authority
  if (domainInfo.isKnownAuthority && !domainInfo.hasAtSymbolUserinfo) {
    safeFactors.push(`Verified reputable domain authority (${domainInfo.hostname})`);
  }

  // 3. Userinfo @ attack
  if (domainInfo.hasAtSymbolUserinfo || features.at_symbol > 0) {
    riskFactors.push('Malicious @ symbol RFC userinfo redirection detected');
  }

  // 4. Sensitive keywords analysis
  if (features.sensitive_words_count > 0) {
    if (domainInfo.isKnownAuthority) {
      safeFactors.push(
        `Authentic account/auth functionality on verified domain (${features.sensitive_words_count} keywords)`
      );
    } else {
      riskFactors.push(
        `Contains ${features.sensitive_words_count} credential keywords (login, verify, account, billing) on unverified domain`
      );
    }
  } else {
    safeFactors.push('Zero credential harvesting trigger keywords');
  }

  // 5. Hyphen typosquatting vs innocent path slugs
  if (domainInfo.domainHyphens >= 2) {
    riskFactors.push(
      `Multiple domain hyphens (${domainInfo.domainHyphens}) indicative of brand typosquatting`
    );
  } else if (features.nb_hyphens < 2) {
    safeFactors.push('Low/normal hyphen count (minimal typosquatting markers)');
  }

  // 6. Multiple www or com token anomalies
  if (features.nb_www > 1) {
    riskFactors.push(`Multiple "www" occurrences (${features.nb_www}) detected`);
  }
  if (features.nb_com > 1) {
    riskFactors.push(`Multiple "com" tokens (${features.nb_com}) found in URL string`);
  }

  // 7. HTTPS encryption scheme
  if (features.isHttps === 1) {
    safeFactors.push('TLS/HTTPS protocol scheme detected');
  } else if (!domainInfo.isKnownAuthority) {
    riskFactors.push('Unencrypted plain HTTP protocol scheme');
  }

  // 8. Dot nesting & subdomain depth
  if (features.nb_dots >= 5) {
    riskFactors.push(
      `Excessive dot delimiters (${features.nb_dots}) indicating deep subdomain nesting`
    );
  } else if (features.nb_dots <= 3) {
    safeFactors.push(`Standard dot count (${features.nb_dots} dots, clean subdomain structure)`);
  }

  // 9. Length factor
  if (features.url_length <= 65) {
    safeFactors.push(`Compact URL length (${features.url_length} chars)`);
  } else if (riskFactors.length > 0 && !domainInfo.isKnownAuthority) {
    riskFactors.push(`Extended URL length (${features.url_length} chars)`);
  }

  // Harmonization synthesis:
  // Legitimate when verified domain without spoofing OR when 0 threat anomalies and >= 3 benign indicators
  const isHarmonizedLegitimate =
    (domainInfo.isKnownAuthority && !domainInfo.hasAtSymbolUserinfo && !domainInfo.isRawIp) ||
    (riskFactors.length === 0 && safeFactors.length >= 3);

  // Confirmed phishing when hard threat vectors are present
  const isConfirmedPhishing =
    domainInfo.hasAtSymbolUserinfo ||
    domainInfo.isRawIp ||
    features.valid_url === 1 ||
    riskFactors.length >= 2;

  let riskLevel: RiskLevel = 'safe';
  if (isHarmonizedLegitimate) {
    riskLevel = 'safe';
  } else if (isConfirmedPhishing || riskFactors.length >= 2) {
    riskLevel = riskFactors.length >= 3 ? 'critical' : 'high';
  } else if (riskFactors.length === 1) {
    riskLevel = 'moderate';
  } else if (phishProb >= 0.25) {
    riskLevel = 'low';
  } else {
    riskLevel = 'safe';
  }

  return {
    riskLevel,
    riskFactors,
    safeFactors,
    isHarmonizedLegitimate,
    isConfirmedPhishing,
  };
}

/**
 * Executes complete prediction workflow across requested model or all 3 models
 */
export function predictUrlLocally(
  rawUrl: string,
  selectedModel: ModelId = 'ridge',
  decisionThreshold: number = 0.5
): PredictionResult {
  const trimmedUrl = (rawUrl || '').trim();
  const features = extractURLFeatures(trimmedUrl);

  // Preliminary evaluation for risk and domain alignment
  const preliminaryRisk = evaluateRisk(features, 0.5, trimmedUrl);

  const primaryModelKey: 'standard' | 'ridge' | 'lasso' =
    selectedModel === 'all' ? 'ridge' : selectedModel;

  const allModelKeys: ('standard' | 'ridge' | 'lasso')[] = ['standard', 'ridge', 'lasso'];
  const allModels: SingleModelPrediction[] = allModelKeys.map((m) =>
    runModelInference(features, m, decisionThreshold, {
      isHarmonizedLegitimate: preliminaryRisk.isHarmonizedLegitimate,
      isConfirmedPhishing: preliminaryRisk.isConfirmedPhishing,
    })
  );

  const primaryPrediction = allModels.find((m) => m.modelId === primaryModelKey)!;
  const finalRisk = evaluateRisk(features, primaryPrediction.probability, trimmedUrl);

  return {
    id: `scan-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
    url: trimmedUrl,
    prediction: primaryPrediction.prediction,
    probability: primaryPrediction.probability,
    confidence: primaryPrediction.confidence,
    modelUsed: primaryPrediction.modelName,
    modelId: primaryModelKey,
    features,
    timestamp: new Date().toISOString(),
    riskLevel: finalRisk.riskLevel,
    riskFactors: finalRisk.riskFactors,
    safeFactors: finalRisk.safeFactors,
    allModels,
  };
}
