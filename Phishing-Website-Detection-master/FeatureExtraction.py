# coding: utf-8

import pandas as pd
from urllib.parse import urlparse
import re
from bs4 import BeautifulSoup
import whois
import urllib.request
import time
import socket
from urllib.error import HTTPError
from datetime import datetime


class FeatureExtraction:
    def __init__(self):
        pass

    def getProtocol(self, url):
        return urlparse(url).scheme

    def getDomain(self, url):
        return urlparse(url).netloc

    def getPath(self, url):
        return urlparse(url).path

    def havingIP(self, url):
        """If the domain part has IP then it is phishing, otherwise legitimate."""
        match = re.search(
            '(([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.'
            '([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\/)|'  # IPv4
            '((0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\/)'  # IPv4 in hex
            '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # IPv6
        return 1 if match else 0  # 1 = phishing, 0 = legitimate

    def long_url(self, url):
        """Classifies URLs based on length."""
        if len(url) < 54:
            return 0  # Legitimate
        elif 54 <= len(url) <= 75:
            return 2  # Suspicious
        else:
            return 1  # Phishing

    def have_at_symbol(self, url):
        """Checks if '@' symbol is present in URL."""
        return 1 if "@" in url else 0  # 1 = phishing, 0 = legitimate

    def redirection(self, url):
        """Checks for '//' in the URL path (indicates possible redirection)."""
        return 1 if "//" in urlparse(url).path else 0  # 1 = phishing, 0 = legitimate

    def prefix_suffix_separation(self, url):
        """Checks if domain contains '-' (common in phishing URLs)."""
        return 1 if "-" in urlparse(url).netloc else 0  # 1 = phishing, 0 = legitimate

    def sub_domains(self, url):
        """Classifies URLs based on the number of subdomains."""
        dot_count = url.count(".")
        if dot_count < 3:
            return 0  # Legitimate
        elif dot_count == 3:
            return 2  # Suspicious
        else:
            return 1  # Phishing

    def shortening_service(self, url):
        """Detects URL shorteners (bit.ly, tinyurl, etc.), commonly used in phishing."""
        match = re.search(
            'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
            'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
            'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
            'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
            'db\.tt|qr\.ae|adf\.ly|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|po\.st|'
            'bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|'
            'prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|'
            'v\.gd|tr\.im|link\.zip\.net', url)
        return 1 if match else 0  # 1 = phishing, 0 = legitimate

    def web_traffic(self, url):
        """This function is disabled because Alexa API no longer works."""
        return 2  # Default "suspicious" value

    def domain_registration_length(self, url):
        """Checks domain registration length."""
        try:
            domain_info = whois.whois(urlparse(url).netloc)
            expiration_date = domain_info.expiration_date
            if expiration_date is None:
                return 1  # Phishing
            today = datetime.today()
            if isinstance(expiration_date, list):
                expiration_date = expiration_date[0]
            registration_length = (expiration_date - today).days
            return 1 if registration_length / 365 <= 1 else 0  # 1 = phishing, 0 = legitimate
        except:
            return 1  # Assume phishing if WHOIS lookup fails

    def age_domain(self, url):
        """Checks the age of the domain."""
        try:
            domain_info = whois.whois(urlparse(url).netloc)
            creation_date = domain_info.creation_date
            if creation_date is None:
                return 1  # Phishing
            today = datetime.today()
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            age = (today - creation_date).days
            return 1 if (age / 30) < 6 else 0  # 1 = phishing, 0 = legitimate
        except:
            return 1  # Assume phishing if WHOIS lookup fails

    def dns_record(self, url):
        """Checks if a DNS record exists."""
        try:
            domain_info = whois.whois(urlparse(url).netloc)
            return 0 if domain_info else 1  # 0 = legitimate, 1 = phishing
        except:
            return 1  # Phishing (if lookup fails)

    def https_token(self, url):
        """Detects misuse of HTTPS in the URL."""
        match = re.search('https://|http://', url)
        try:
            if match.start(0) == 0:
                url = url[match.end(0):]
                return 1 if re.search('http|https', url) else 0
        except:
            return 1
        return 0


def getAttributess(url):
    """Extracts all features from a given URL."""
    fe = FeatureExtraction()
    features = {
        'Protocol': fe.getProtocol(url),
        'Domain': fe.getDomain(url),
        'Path': fe.getPath(url),
        'Having_IP': fe.havingIP(url),
        'URL_Length': fe.long_url(url),
        'Having_@_symbol': fe.have_at_symbol(url),
        'Redirection_//_symbol': fe.redirection(url),
        'Prefix_suffix_separation': fe.prefix_suffix_separation(url),
        'Sub_domains': fe.sub_domains(url),
        'Tiny_URL': fe.shortening_service(url),
        'Web_Traffic': fe.web_traffic(url),
        'Domain_Registration_Length': fe.domain_registration_length(url),
        'DNS_Record': fe.dns_record(url),
        'Age_Domain': fe.age_domain(url),
        'HTTPS_Token': fe.https_token(url)
    }
    return pd.DataFrame([features])
