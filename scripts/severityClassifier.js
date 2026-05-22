/**
 * Automated Severity Classification Service
 * Classifies IOCs based on type, source, and context
 */

const { MetricTimer } = require("./metricsCollector");

const logger = console;

// Source reliability scoring
const SOURCE_SCORES = {
  // High reliability sources (80-100)
  'urlhaus': 90,
  'threatfox': 90,
  'otx': 85,
  'emergingthreats': 85,
  'spamhaus': 90,
  'ciarmy': 80,
  
  // Medium-high reliability (60-79)
  'phishtank': 75,
  'phishstats': 75,
  'bazaar': 70,
  'malshare': 75,
  'misp': 80,
  'ipsum': 78,
  'c2intel': 78,
  'yaraforge': 80,
  'nvd': 88,
  
  // Medium reliability (40-59)
  'dshield_openioc': 60,
  'dshield_threatfeeds': 60,
  'bazaar_yara': 65,
  
  // Default for unknown sources
  'default': 50
};

// Type-based threat severity
const TYPE_SEVERITY_MAP = {
  'hash': { base: 'high', score: 80 },           // Malware hashes are serious
  'ip': { base: 'medium', score: 60 },           // IPs can be dynamic
  'domain': { base: 'medium', score: 65 },       // Domains moderate risk
  'url': { base: 'high', score: 75 },            // URLs often indicate active threats
  'cve': { base: 'high', score: 82 },            // CVEs can indicate exploitable vulnerabilities
  'email': { base: 'medium', score: 60 },        // Email addresses medium risk
  'file': { base: 'high', score: 80 },           // File IOCs are serious
  'default': { base: 'medium', score: 50 }
};

// Threat keywords that increase severity
const CRITICAL_KEYWORDS = [
  'ransomware', 'apt', 'advanced persistent', 'zero-day', 'exploit',
  'backdoor', 'trojan', 'rat', 'remote access', 'c2', 'command and control',
  'cryptominer', 'miner', 'botnet', 'ddos'
];

const HIGH_KEYWORDS = [
  'malware', 'phishing', 'phish', 'scam', 'fraud', 'stealer',
  'banking', 'credential', 'keylogger', 'spyware', 'adware'
];

const MEDIUM_KEYWORDS = [
  'suspicious', 'potentially unwanted', 'pua', 'pup', 'unwanted'
];

/**
 * Calculate severity score (0-100)
 * @param {Object} ioc - The IOC object with type, value, source, description, etc.
 * @returns {number} - Severity score 0-100
 */
function calculateSeverityScore(ioc) {
  let score = 50; // Start with neutral score
  
  // Factor 1: Source reliability (30% weight)
  const sourceKey = (ioc.source || '').toLowerCase().split('_')[0];
  const sourceScore = SOURCE_SCORES[sourceKey] || SOURCE_SCORES['default'];
  score += (sourceScore - 50) * 0.3;
  
  // Factor 2: Type-based severity (30% weight)
  const typeInfo = TYPE_SEVERITY_MAP[ioc.type] || TYPE_SEVERITY_MAP['default'];
  score += (typeInfo.score - 50) * 0.3;
  
  // Factor 3: Description/context analysis (40% weight)
  const description = (ioc.description || '').toLowerCase();
  const tags = Array.isArray(ioc.tags) ? ioc.tags.join(' ').toLowerCase() : '';
  const combinedText = `${description} ${tags}`;
  
  if (CRITICAL_KEYWORDS.some(kw => combinedText.includes(kw))) {
    score += 20; // Significant boost for critical threats
  } else if (HIGH_KEYWORDS.some(kw => combinedText.includes(kw))) {
    score += 10; // Moderate boost for high threats
  } else if (MEDIUM_KEYWORDS.some(kw => combinedText.includes(kw))) {
    score += 5; // Small boost for medium threats
  }
  
  // Factor 4: Observation count boost (multi-source corroboration)
  if (ioc.observedCount && ioc.observedCount > 1) {
    const observationBoost = Math.min(15, Math.log(ioc.observedCount) * 4);
    score += observationBoost;
  }
  
  // Factor 5: Confidence adjustment
  if (ioc.confidence) {
    const confidenceAdjust = (ioc.confidence - 50) * 0.1;
    score += confidenceAdjust;
  }
  
  // Clamp score between 0-100
  return Math.max(0, Math.min(100, Math.round(score)));
}

/**
 * Convert severity score to severity level
 * @param {number} score - Severity score 0-100
 * @returns {string} - Severity level: critical, high, medium, low, info
 */
function scoreToSeverity(score) {
  if (score >= 85) return 'critical';
  if (score >= 70) return 'high';
  if (score >= 50) return 'medium';
  if (score >= 30) return 'low';
  return 'info';
}

/**
 * Calculate confidence level based on source and data quality
 * @param {Object} ioc - The IOC object
 * @returns {number} - Confidence score 0-100
 */
function calculateConfidence(ioc) {
  let confidence = 50; // Base confidence
  
  // Source reliability contributes to confidence
  const sourceKey = (ioc.source || '').toLowerCase().split('_')[0];
  const sourceScore = SOURCE_SCORES[sourceKey] || SOURCE_SCORES['default'];
  confidence = sourceScore;
  
  // Boost confidence if we have rich metadata
  if (ioc.description && ioc.description.length > 20) {
    confidence += 5;
  }
  
  if (ioc.tags && Array.isArray(ioc.tags) && ioc.tags.length > 0) {
    confidence += 5;
  }
  
  // Multiple observations increase confidence
  if (ioc.observedCount && ioc.observedCount > 3) {
    confidence += Math.min(10, ioc.observedCount * 2);
  }
  
  // Clamp confidence between 0-100
  return Math.max(0, Math.min(100, Math.round(confidence)));
}

/**
 * Classify IOC severity and confidence
 * @param {Object} ioc - The IOC object to classify
 * @returns {Object} - { severity, severityScore, confidence }
 */
function classifyIOC(ioc) {
  try {
    // Calculate severity score
    const severityScore = calculateSeverityScore(ioc);
    const severity = scoreToSeverity(severityScore);
    
    // Calculate confidence if not already present
    const confidence = ioc.confidence || calculateConfidence(ioc);
    
    return {
      severity,
      severityScore,
      confidence
    };
  } catch (error) {
    logger.error(`[SeverityClassifier] Error classifying IOC: ${error.message}`);
    return {
      severity: 'medium',
      severityScore: 50,
      confidence: 50
    };
  }
}

/**
 * Batch classify multiple IOCs
 * @param {Array} iocs - Array of IOC objects
 * @returns {Array} - Array of classified IOCs with severity and confidence
 */
async function batchClassify(iocs) {
  // ✅ Metrics: Start scoring timer
  const metricTimer = new MetricTimer('batch_classify', 'scoring', 'severity_classifier');
  
  const classified = iocs.map(ioc => {
    const classification = classifyIOC(ioc);
    return {
      ...ioc,
      ...classification
    };
  });
  
  // ✅ Metrics: Record results
  metricTimer.recordItems(iocs.length, 0);
  metricTimer.addMetadata('avgSeverityScore', classified.reduce((sum, ioc) => sum + (ioc.severityScore || 0), 0) / classified.length);
  await metricTimer.end();
  
  return classified;
}

/**
 * Get severity statistics from a set of IOCs
 * @param {Array} iocs - Array of IOC objects with severity
 * @returns {Object} - Statistics object
 */
function getSeverityStats(iocs) {
  const stats = {
    total: iocs.length,
    bySeverity: {
      critical: 0,
      high: 0,
      medium: 0,
      low: 0,
      info: 0
    },
    averageScore: 0,
    averageConfidence: 0
  };
  
  let totalScore = 0;
  let totalConfidence = 0;
  
  iocs.forEach(ioc => {
    if (ioc.severity) {
      stats.bySeverity[ioc.severity]++;
    }
    if (ioc.severityScore) {
      totalScore += ioc.severityScore;
    }
    if (ioc.confidence) {
      totalConfidence += ioc.confidence;
    }
  });
  
  stats.averageScore = iocs.length > 0 ? Math.round(totalScore / iocs.length) : 0;
  stats.averageConfidence = iocs.length > 0 ? Math.round(totalConfidence / iocs.length) : 0;
  
  return stats;
}

module.exports = {
  classifyIOC,
  batchClassify,
  calculateSeverityScore,
  scoreToSeverity,
  calculateConfidence,
  getSeverityStats,
  SOURCE_SCORES,
  TYPE_SEVERITY_MAP
};
