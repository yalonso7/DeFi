#!/usr/bin/env python3
"""
Blockchain Transaction Network Traffic Monitor
Monitors network traffic for blockchain-related communications
"""

import socket
import struct
import time
import json
import hashlib
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import threading
import queue

# Common blockchain ports
BLOCKCHAIN_PORTS = {
    8333: "Bitcoin",
    8332: "Bitcoin RPC",
    18333: "Bitcoin Testnet",
    9333: "Dogecoin",
    8545: "Ethereum JSON-RPC",
    30303: "Ethereum P2P",
    9050: "Tor SOCKS",
    9051: "Tor Control",
    26656: "Cosmos",
    26657: "Cosmos RPC"
}

@dataclass
class BlockchainTransaction:
    timestamp: str
    source_ip: str
    dest_ip: str
    source_port: int
    dest_port: int
    protocol: str
    blockchain_type: str
    packet_size: int
    tx_hash: str
    confidence: float
    tor_related: bool
    
    def to_dict(self) -> Dict:
        return asdict(self)

class BlockchainTrafficMonitor:
    def __init__(self, interface='eth0', output_file='blockchain_traffic.json'):
        self.interface = interface
        self.output_file = output_file
        self.transactions = []
        self.stats = defaultdict(int)
        self.running = False
        self.packet_queue = queue.Queue()
        
    def parse_ip_header(self, data: bytes) -> Dict[str, Any]:
        """Parse IP header from raw packet data"""
        if len(data) < 20:
            return None
            
        ip_header = struct.unpack('!BBHHHBBH4s4s', data[:20])
        
        version_ihl = ip_header[0]
        version = version_ihl >> 4
        ihl = version_ihl & 0xF
        iph_length = ihl * 4
        
        protocol = ip_header[6]
        src_addr = socket.inet_ntoa(ip_header[8])
        dest_addr = socket.inet_ntoa(ip_header[9])
        
        return {
            'version': version,
            'ihl': ihl,
            'header_length': iph_length,
            'protocol': protocol,
            'src_ip': src_addr,
            'dest_ip': dest_addr,
            'data': data[iph_length:]
        }
    
    def parse_tcp_header(self, data: bytes) -> Dict[str, Any]:
        """Parse TCP header"""
        if len(data) < 20:
            return None
            
        tcp_header = struct.unpack('!HHLLBBHHH', data[:20])
        
        src_port = tcp_header[0]
        dest_port = tcp_header[1]
        sequence = tcp_header[2]
        acknowledgement = tcp_header[3]
        
        doff_reserved = tcp_header[4]
        tcph_length = (doff_reserved >> 4) * 4
        
        return {
            'src_port': src_port,
            'dest_port': dest_port,
            'sequence': sequence,
            'acknowledgement': acknowledgement,
            'header_length': tcph_length,
            'data': data[tcph_length:]
        }
    
    def is_tor_traffic(self, ip: str, port: int) -> bool:
        """Check if traffic is Tor-related"""
        # Check for Tor ports
        if port in [9050, 9051, 9001, 9030]:
            return True
        
        # Check for known Tor exit node IPs (simplified - would need real list)
        # In practice, you'd maintain a list of Tor exit nodes
        return False
    
    def detect_blockchain_protocol(self, payload: bytes, src_port: int, dest_port: int) -> tuple:
        """Detect blockchain protocol from packet payload"""
        if len(payload) < 4:
            return None, 0.0
        
        confidence = 0.0
        blockchain_type = "Unknown"
        
        # Bitcoin magic bytes
        bitcoin_magic = {
            b'\xf9\xbe\xb4\xd9': ('Bitcoin Mainnet', 0.95),
            b'\x0b\x11\x09\x07': ('Bitcoin Testnet', 0.95),
            b'\xfa\xbf\xb5\xda': ('Bitcoin Regtest', 0.95)
        }
        
        # Check for Bitcoin magic bytes
        for magic, (name, conf) in bitcoin_magic.items():
            if payload.startswith(magic):
                return name, conf
        
        # Ethereum RLP encoding detection (simplified)
        if payload[0] in [0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff]:
            if dest_port == 30303 or src_port == 30303:
                return "Ethereum P2P", 0.85
        
        # JSON-RPC detection for Ethereum
        if b'"jsonrpc"' in payload[:200] and (dest_port == 8545 or src_port == 8545):
            return "Ethereum JSON-RPC", 0.90
        
        # Port-based detection
        if dest_port in BLOCKCHAIN_PORTS:
            return BLOCKCHAIN_PORTS[dest_port], 0.60
        elif src_port in BLOCKCHAIN_PORTS:
            return BLOCKCHAIN_PORTS[src_port], 0.60
        
        # Pattern matching for common blockchain strings
        blockchain_patterns = [
            (b'version', 0.3),
            (b'verack', 0.3),
            (b'getaddr', 0.4),
            (b'addr', 0.3),
            (b'inv', 0.3),
            (b'getdata', 0.4),
            (b'block', 0.4),
            (b'tx', 0.3),
        ]
        
        for pattern, weight in blockchain_patterns:
            if pattern in payload[:100]:
                confidence += weight
        
        if confidence > 0.5:
            blockchain_type = "Likely Blockchain Traffic"
        
        return blockchain_type, min(confidence, 1.0)
    
    def generate_tx_hash(self, packet_data: Dict) -> str:
        """Generate a unique hash for the transaction"""
        hash_input = f"{packet_data['timestamp']}{packet_data['src_ip']}{packet_data['dest_ip']}{packet_data['sequence']}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def process_packet(self, packet: bytes):
        """Process a single packet"""
        try:
            ip_data = self.parse_ip_header(packet)
            if not ip_data:
                return
            
            # Only process TCP packets (protocol 6)
            if ip_data['protocol'] != 6:
                return
            
            tcp_data = self.parse_tcp_header(ip_data['data'])
            if not tcp_data:
                return
            
            # Check if Tor-related
            tor_related = (
                self.is_tor_traffic(ip_data['src_ip'], tcp_data['src_port']) or
                self.is_tor_traffic(ip_data['dest_ip'], tcp_data['dest_port'])
            )
            
            # Detect blockchain protocol
            blockchain_type, confidence = self.detect_blockchain_protocol(
                tcp_data['data'],
                tcp_data['src_port'],
                tcp_data['dest_port']
            )
            
            # Only log if you have reasonable confidence it's blockchain-related
            if confidence > 0.5 or tor_related:
                packet_info = {
                    'timestamp': datetime.now().isoformat(),
                    'src_ip': ip_data['src_ip'],
                    'dest_ip': ip_data['dest_ip'],
                    'src_port': tcp_data['src_port'],
                    'dest_port': tcp_data['dest_port'],
                    'sequence': tcp_data['sequence']
                }
                
                transaction = BlockchainTransaction(
                    timestamp=packet_info['timestamp'],
                    source_ip=packet_info['src_ip'],
                    dest_ip=packet_info['dest_ip'],
                    source_port=packet_info['src_port'],
                    dest_port=packet_info['dest_port'],
                    protocol='TCP',
                    blockchain_type=blockchain_type,
                    packet_size=len(packet),
                    tx_hash=self.generate_tx_hash(packet_info),
                    confidence=confidence,
                    tor_related=tor_related
                )
                
                self.transactions.append(transaction)
                self.stats[blockchain_type] += 1
                self.stats['total_packets'] += 1
                if tor_related:
                    self.stats['tor_related'] += 1
                
                print(f"[{transaction.timestamp}] {blockchain_type} | "
                      f"{transaction.source_ip}:{transaction.source_port} -> "
                      f"{transaction.dest_ip}:{transaction.dest_port} | "
                      f"Confidence: {confidence:.2f} | Tor: {tor_related}")
                
        except Exception as e:
            print(f"Error processing packet: {e}")
    
    def packet_capture_thread(self, sock):
        """Thread for capturing packets"""
        while self.running:
            try:
                packet, addr = sock.recvfrom(65565)
                self.packet_queue.put(packet)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Capture error: {e}")
                break
    
    def packet_processing_thread(self):
        """Thread for processing packets"""
        while self.running:
            try:
                packet = self.packet_queue.get(timeout=1)
                self.process_packet(packet)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
    
    def start_monitoring(self, duration=60):
        """Start monitoring network traffic"""
        print(f"Starting blockchain traffic monitor...")
        print(f"Monitoring for {duration} seconds")
        print(f"Looking for blockchain protocols on known ports: {list(BLOCKCHAIN_PORTS.values())}")
        print("-" * 80)
        
        try:
            # Create raw socket (requires root/admin privileges)
            sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_TCP)
            sock.settimeout(1.0)
            
            self.running = True
            
            # Start threads
            capture_thread = threading.Thread(target=self.packet_capture_thread, args=(sock,))
            processing_threads = [
                threading.Thread(target=self.packet_processing_thread)
                for _ in range(2)  # Use 2 processing threads
            ]
            
            capture_thread.start()
            for pt in processing_threads:
                pt.start()
            
            # Monitor for specified duration
            time.sleep(duration)
            
            self.running = False
            
            # Wait for threads to finish
            capture_thread.join()
            for pt in processing_threads:
                pt.join()
            
            sock.close()
            
        except PermissionError:
            print("Error: Raw socket requires root/administrator privileges")
            print("Run with: sudo python3 script.py")
            return
        except Exception as e:
            print(f"Error: {e}")
            return
        
        self.generate_report()
    
    def generate_report(self):
        """Generate detailed report of captured transactions"""
        print("\n" + "=" * 80)
        print("BLOCKCHAIN TRAFFIC ANALYSIS REPORT")
        print("=" * 80)
        
        print(f"\nTotal Packets Analyzed: {self.stats['total_packets']}")
        print(f"Tor-Related Traffic: {self.stats['tor_related']}")
        
        print("\n--- Blockchain Protocol Distribution ---")
        blockchain_stats = {k: v for k, v in self.stats.items() 
                           if k not in ['total_packets', 'tor_related']}
        for protocol, count in sorted(blockchain_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"{protocol}: {count} packets")
        
        # Save to JSON
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'statistics': dict(self.stats),
            'transactions': [tx.to_dict() for tx in self.transactions[-100:]]  # Last 100
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📊 Full report saved to: {self.output_file}")
        
        if self.transactions:
            print("\n--- Sample Transactions ---")
            for tx in self.transactions[:5]:
                print(f"\nTransaction: {tx.tx_hash}")
                print(f"  Time: {tx.timestamp}")
                print(f"  Route: {tx.source_ip}:{tx.source_port} -> {tx.dest_ip}:{tx.dest_port}")
                print(f"  Type: {tx.blockchain_type}")
                print(f"  Confidence: {tx.confidence:.2%}")
                print(f"  Tor: {tx.tor_related}")
                print(f"  Size: {tx.packet_size} bytes")

if __name__ == "__main__":
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║     Blockchain Transaction Network Traffic Monitor           ║
║              Tor Network Analysis Tool                       ║
╚══════════════════════════════════════════════════════════════╝

⚠️  WARNING: This tool requires root/administrator privileges
⚠️  Only use on networks you own or have permission to monitor
⚠️  Respect privacy and legal requirements

    """)
    
    duration = 60
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            print("Usage: python3 script.py [duration_in_seconds]")
            sys.exit(1)
    
    monitor = BlockchainTrafficMonitor()
    monitor.start_monitoring(duration=duration) 763-2304
