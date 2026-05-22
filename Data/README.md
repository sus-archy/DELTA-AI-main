# Data Folder

Place your IOC database file in this folder.

## Required: Main Database

**`DB-ThreatIndicators.csv`** — The only required file. Place it with this exact name.

Expected columns:
- `type` — IOC category (url, domain, ip, hash, etc.)
- `value` — The IOC value
- `severity` — Ground truth label: `medium`, `high`, or `critical` (lowercase)
- `source` — Where the indicator originated
- `description` — Human-readable description
- `tags` — JSON array as string, e.g. `["ransomware", "c2"]`
- `firstSeen` — ISO 8601 datetime
- `lastSeen` — ISO 8601 datetime
- `observedCount` — Integer count of observations
- `raw` — JSON string (can be `{}`)
- `confidence` — Float 0-1 (optional)

**Minimal CSV example:**
```csv
type,value,severity,source,description,tags,firstSeen,lastSeen,observedCount,raw,confidence
url,http://malware.com,high,urlhaus,Ransomware site,"[""ransomware""]",2024-01-01T00:00:00Z,2024-03-20T00:00:00Z,245,{},0.95
domain,c2server.xyz,critical,otx,C2 infrastructure,"[""c2"",""botnet""]",2024-01-15T00:00:00Z,2024-03-20T00:00:00Z,1200,{},0.99
ip,1.2.3.4,medium,spamhaus,Spam source,"[""spam""]",2024-02-01T00:00:00Z,2024-03-20T00:00:00Z,50,{},0.85
```

## Optional: Enrichment Sidecars

These files are optional. The pipeline works with just the main CSV.

**`merged_ip_list.txt`** — Known bad IP addresses and CIDR ranges.
```
192.168.1.100
10.0.0.0/8
8.8.8.8/24
```

**`merged_phishing_data.csv`** — Phishing domain database.
```
domain,host,date,target
evil-phishing.com,fake-login,2024-01-01,paypal
```
If these files are not present or are set to `null` in your config, the pipeline skips enrichment.