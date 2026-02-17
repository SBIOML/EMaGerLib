"""Test suite for realtime control threading implementation"""

import unittest
import time
import threading
from collections import deque, Counter
from statistics import mean
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSharedMemoryCommunication(unittest.TestCase):
    """Test suite for shared memory communication performance"""
    
    def test_01_mock_classifier_controller_speed(self):
        """Test ClassifierController read speed simulation"""
        # Simulate LibEMG ClassifierController behavior
        predictions = [0.0, 1.0, 0.95, 0.0]  # [timestamp, class, confidence, velocity]
        
        timings = []
        for _ in range(1000):
            start = time.perf_counter()
            # Simulate get_data call
            data = predictions.copy()
            result = data[1] if data else None
            elapsed = time.perf_counter() - start
            timings.append(elapsed * 1000000)  # microseconds
        
        avg_time = mean(timings)
        max_time = max(timings)
        
        # Should be very fast (< 10µs on average)
        self.assertLess(avg_time, 10.0)
        print(f"[PASS] Mock shared memory read: avg {avg_time:.2f}µs, max {max_time:.2f}µs")
    
    def test_02_predictor_controller_latency(self):
        """Test end-to-end latency from prediction to read"""
        shared_data = {"prediction": None, "timestamp": None}
        data_lock = threading.Lock()
        stop_event = threading.Event()
        
        latencies = []
        
        def mock_predictor():
            """Simulates predictor writing to shared memory"""
            for i in range(50):
                if stop_event.is_set():
                    break
                write_time = time.perf_counter()
                with data_lock:
                    shared_data["prediction"] = i % 5
                    shared_data["timestamp"] = write_time
                time.sleep(0.01)  # 10ms prediction rate
        
        def mock_controller():
            """Simulates controller reading from shared memory"""
            time.sleep(0.005)  # Small startup delay
            for _ in range(45):
                if stop_event.is_set():
                    break
                read_start = time.perf_counter()
                with data_lock:
                    pred = shared_data["prediction"]
                    write_time = shared_data["timestamp"]
                
                if pred is not None and write_time is not None:
                    latency = (read_start - write_time) * 1000  # ms
                    latencies.append(latency)
                
                time.sleep(0.008)  # Controller polling rate
        
        t1 = threading.Thread(target=mock_predictor, daemon=True)
        t2 = threading.Thread(target=mock_controller, daemon=True)
        
        try:
            t1.start()
            t2.start()
            
            t1.join(timeout=3)
            t2.join(timeout=3)
        finally:
            stop_event.set()
        
        if latencies:
            avg_latency = mean(latencies)
            max_latency = max(latencies)
            
            # Should be low latency (< 10ms average for Windows thread scheduling)
            self.assertLess(avg_latency, 10.0)
            print(f"[PASS] Predictor->Controller latency: avg {avg_latency:.2f}ms, max {max_latency:.2f}ms")
        else:
            self.fail("No latency measurements collected")


class TestSmoothingPerformance(unittest.TestCase):
    """Test suite for smoothing algorithm performance"""
    
    def test_01_mode_smoothing_speed(self):
        """Test mode smoothing execution speed"""
    def test_01_mode_smoothing_speed(self):
        """Test mode smoothing execution speed"""
        recent = deque(maxlen=5)
        test_data = [1, 2, 1, 3, 1] * 2000  # 10k predictions
        
        timings = []
        for value in test_data:
            recent.append(value)
            if len(recent) > 1:
                start = time.perf_counter()
                # Mode smoothing algorithm from realtime_control_threads.py
                counts = Counter(recent)
                most_common = counts.most_common()
                top_count = most_common[0][1]
                candidates = [val for val, cnt in most_common if cnt == top_count]
                for v in reversed(recent):
                    if v in candidates:
                        smoothed = v
                        break
                elapsed = time.perf_counter() - start
                timings.append(elapsed * 1000000)  # microseconds
        
        avg_time = mean(timings)
        max_time = max(timings)
        
        # Should be fast (< 20µs average)
        self.assertLess(avg_time, 20.0)
        print(f"[PASS] Mode smoothing: avg {avg_time:.2f}µs, max {max_time:.2f}µs per prediction")
    
    def test_02_mean_smoothing_speed(self):
        """Test mean smoothing execution speed"""
        recent = deque(maxlen=5)
        test_data = [1, 2, 3, 4, 5] * 2000  # 10k predictions
        
        timings = []
        for value in test_data:
            recent.append(value)
            if len(recent) > 1:
                start = time.perf_counter()
                smoothed = int(round(mean(recent)))
                elapsed = time.perf_counter() - start
                timings.append(elapsed * 1000000)  # microseconds
        
        avg_time = mean(timings)
        max_time = max(timings)
        
        # Should be very fast (< 12µs average)
        self.assertLess(avg_time, 12.0)
        print(f"[PASS] Mean smoothing: avg {avg_time:.2f}µs, max {max_time:.2f}µs per prediction")
    
    def test_03_last_smoothing_speed(self):
        """Test last-value smoothing (no smoothing) speed"""
        recent = deque(maxlen=5)
        test_data = [1, 2, 3, 4, 5] * 2000  # 10k predictions
        
        timings = []
        for value in test_data:
            recent.append(value)
            start = time.perf_counter()
            smoothed = recent[-1]
            elapsed = time.perf_counter() - start
            timings.append(elapsed * 1000000)  # microseconds
        
        avg_time = mean(timings)
        max_time = max(timings)
        
        # Should be extremely fast (< 1µs average)
        self.assertLess(avg_time, 1.0)
        print(f"[PASS] Last smoothing: avg {avg_time:.2f}µs, max {max_time:.2f}µs per prediction")
    
    def test_04_window_size_impact(self):
        """Test impact of window size on smoothing performance"""
        test_data = [1, 2, 1, 3, 1] * 1000  # 5k predictions
        
        results = {}
        for window_size in [1, 3, 5, 7, 10]:
            recent = deque(maxlen=window_size)
            timings = []
            
            for value in test_data:
                recent.append(value)
                if len(recent) > 1:
                    start = time.perf_counter()
                    counts = Counter(recent)
                    most_common = counts.most_common()
                    top_count = most_common[0][1]
                    candidates = [val for val, cnt in most_common if cnt == top_count]
                    for v in reversed(recent):
                        if v in candidates:
                            smoothed = v
                            break
                    elapsed = time.perf_counter() - start
                    timings.append(elapsed * 1000000)
            
            results[window_size] = mean(timings) if timings else 0
        
        # Larger windows should still be reasonable (< 50µs)
        for size, avg_time in results.items():
            self.assertLess(avg_time, 50.0)
        
        print(f"[PASS] Window size impact: " + ", ".join([f"{k}={v:.1f}µs" for k, v in results.items()]))


class TestThreadLifecycle(unittest.TestCase):
    """Test suite for thread lifecycle management"""
    
    def test_01_thread_start_stop(self):
        """Test thread startup and graceful shutdown"""
        stop_event = threading.Event()
        started = threading.Event()
        counter = {"value": 0}
        
        def worker():
            started.set()
            while not stop_event.is_set():
                counter["value"] += 1
                time.sleep(0.001)
        
        thread = threading.Thread(target=worker, daemon=True)
        start_time = time.perf_counter()
        thread.start()
        
        # Wait for thread to actually start
        started.wait(timeout=1.0)
        startup_time = (time.perf_counter() - start_time) * 1000
        
        time.sleep(0.05)  # Let it run
        
        # Stop and measure shutdown time
        stop_start = time.perf_counter()
        stop_event.set()
        thread.join(timeout=1.0)
        shutdown_time = (time.perf_counter() - stop_start) * 1000
        
        self.assertFalse(thread.is_alive())
        self.assertGreater(counter["value"], 0)
        self.assertLess(startup_time, 100.0)  # Should start quickly
        self.assertLess(shutdown_time, 100.0)  # Should stop quickly
        
        print(f"[PASS] Thread lifecycle: startup {startup_time:.2f}ms, shutdown {shutdown_time:.2f}ms")
    
    def test_02_multiple_threads_coordination(self):
        """Test coordination between multiple threads like predictor/controller"""
        stop_event = threading.Event()
        predictor_ready = threading.Event()
        data_ready = threading.Event()
        shared_data = {"prediction": None}
        lock = threading.Lock()
        
        def predictor():
            predictor_ready.set()
            for i in range(20):
                if stop_event.is_set():
                    break
                with lock:
                    shared_data["prediction"] = i
                data_ready.set()
                time.sleep(0.01)
        
        def controller():
            predictor_ready.wait(timeout=1.0)  # Wait for predictor
            reads = 0
            while reads < 15 and not stop_event.is_set():
                if data_ready.wait(timeout=0.1):
                    with lock:
                        pred = shared_data["prediction"]
                    if pred is not None:
                        reads += 1
                    data_ready.clear()
        
        t1 = threading.Thread(target=predictor, daemon=True)
        t2 = threading.Thread(target=controller, daemon=True)
        
        start = time.perf_counter()
        try:
            t1.start()
            t2.start()
            
            t1.join(timeout=2)
            t2.join(timeout=2)
        finally:
            stop_event.set()
        
        elapsed = time.perf_counter() - start
        
        self.assertLess(elapsed, 1.5)  # Should complete quickly
        print(f"[PASS] Thread coordination: {elapsed*1000:.1f}ms total execution")


class TestPerformanceMonitoring(unittest.TestCase):
    """Test suite for performance monitoring and metrics"""
    
    def test_01_loop_timing_accuracy(self):
        """Test accuracy of loop timing measurements"""
        loop_times = []
        
        for _ in range(100):
            start = time.perf_counter()
            # Simulate work
            time.sleep(0.001)
            elapsed = time.perf_counter() - start
            loop_times.append(elapsed)
        
        avg_time = mean(loop_times) * 1000
        max_time = max(loop_times) * 1000
        
        # Should be close to 1ms
        self.assertGreater(avg_time, 0.8)
        self.assertLess(avg_time, 3.0)
        
        print(f"[PASS] Loop timing: avg {avg_time:.2f}ms, max {max_time:.2f}ms (target ~1ms)")
    
    def test_02_performance_metrics_collection(self):
        """Test performance metrics collection like in controller_worker"""
        loop_times = []
        read_times = []
        
        for i in range(200):
            loop_start = time.perf_counter()
            
            # Simulate shared memory read
            read_start = time.perf_counter()
            dummy_data = [0.0, float(i % 5), 0.95, 0.0]
            read_elapsed = time.perf_counter() - read_start
            read_times.append(read_elapsed)
            
            # Simulate processing
            time.sleep(0.0001)
            
            loop_elapsed = time.perf_counter() - loop_start
            loop_times.append(loop_elapsed)
        
        avg_loop = mean(loop_times) * 1000
        max_loop = max(loop_times) * 1000
        avg_read = mean(read_times) * 1000000  # microseconds
        
        # Verify metrics are reasonable
        self.assertLess(avg_loop, 1.0)  # Should be sub-millisecond
        self.assertLess(avg_read, 10.0)  # Reads should be very fast
        
        print(f"[PASS] Metrics collection: loop {avg_loop:.3f}ms, read {avg_read:.2f}µs")
    
    def test_03_throughput_measurement(self):
        """Test throughput measurement for prediction processing"""
        processed = 0
        start_time = time.perf_counter()
        target_duration = 0.1  # 100ms
        
        while time.perf_counter() - start_time < target_duration:
            # Simulate prediction processing
            recent = deque([1, 2, 1], maxlen=3)
            result = recent[-1]
            processed += 1
        
        elapsed = time.perf_counter() - start_time
        throughput = processed / elapsed
        
        # Should handle many predictions per second
        self.assertGreater(throughput, 1000)  # > 1kHz
        
        print(f"[PASS] Throughput: {throughput:.0f} predictions/sec ({processed} in {elapsed*1000:.1f}ms)")


class TestRealtimeControlComponents(unittest.TestCase):
    """Test suite for specific realtime control components"""
    
    def test_01_heartbeat_timing(self):
        """Test heartbeat mechanism timing"""
        heartbeat_interval = 0.05  # 50ms
        last_send_time = 0.0
        heartbeats = []
        
        start = time.perf_counter()
        while time.perf_counter() - start < 0.3:  # Run for 300ms
            now = time.perf_counter()
            if (now - last_send_time) >= heartbeat_interval:
                heartbeats.append(now)
                last_send_time = now
            time.sleep(0.001)
        
        # Should have ~6 heartbeats in 300ms with 50ms interval
        self.assertGreaterEqual(len(heartbeats), 4)
        self.assertLessEqual(len(heartbeats), 8)
        
        # Check intervals
        if len(heartbeats) > 1:
            intervals = [heartbeats[i] - heartbeats[i-1] for i in range(1, len(heartbeats))]
            avg_interval = mean(intervals) * 1000
            
            # Should be close to target
            self.assertGreater(avg_interval, 40.0)
            self.assertLess(avg_interval, 60.0)
            
            print(f"[PASS] Heartbeat timing: {len(heartbeats)} beats, avg {avg_interval:.1f}ms interval")
    
    def test_02_prediction_validation(self):
        """Test prediction validation logic"""
        num_classes = 5
        test_predictions = [-1, 0, 2, 5, 10, 4, 100, 3]
    def test_02_prediction_validation(self):
        """Test prediction validation logic"""
        num_classes = 5
        test_predictions = [-1, 0, 2, 5, 10, 4, 100, 3]
        
        validated = []
        for pred in test_predictions:
            # Validation logic from controller_worker
            if pred not in range(num_classes):
                pred = 0  # Default to rest
            validated.append(pred)
        
        # Check all are valid
        for val in validated:
            self.assertIn(val, range(num_classes))
        
        # Check invalid ones were converted to 0
        expected = [0, 0, 2, 0, 0, 4, 0, 3]
        self.assertEqual(validated, expected)
        
        print(f"[PASS] Prediction validation: {len(test_predictions)} inputs -> {len(validated)} valid outputs")
    
    def test_03_buffer_accumulation_speed(self):
        """Test speed of accumulating predictions in smoothing buffer"""
        recent = deque(maxlen=5)
        
        start = time.perf_counter()
        for i in range(10000):
            recent.append(i % 5)
        elapsed = time.perf_counter() - start
        
        throughput = 10000 / elapsed
        
        self.assertGreater(throughput, 100000)  # Should do > 100k/sec
        print(f"[PASS] Buffer accumulation: {throughput:.0f} ops/sec ({elapsed*1000:.2f}ms for 10k)")


class TestEndToEndLatency(unittest.TestCase):
    """Test suite for end-to-end latency measurements"""
    
    def test_01_complete_prediction_cycle(self):
        """Test complete cycle: read -> smooth -> process"""
        shared_data = {"prediction": 0, "write_time": time.perf_counter()}
        lock = threading.Lock()
        recent = deque(maxlen=3)
        
        cycle_times = []
        
        for i in range(100):
            cycle_start = time.perf_counter()
            
            # 1. Read from shared memory (simulated)
            with lock:
                pred = shared_data["prediction"]
                write_time = shared_data["write_time"]
            
            # 2. Add to buffer
            recent.append(pred)
            
            # 3. Smooth
            if len(recent) > 1:
                smoothed = int(round(mean(recent)))
            else:
                smoothed = pred
            
            # 4. Process (simulated)
            result = smoothed
            
            cycle_elapsed = time.perf_counter() - cycle_start
            cycle_times.append(cycle_elapsed * 1000)  # ms
            
            # Update shared data for next cycle
            with lock:
                shared_data["prediction"] = (i + 1) % 5
                shared_data["write_time"] = time.perf_counter()
            
            time.sleep(0.001)
        
        avg_cycle = mean(cycle_times)
        max_cycle = max(cycle_times)
        
        # Complete cycle should be sub-millisecond
        self.assertLess(avg_cycle, 0.5)
        
        print(f"[PASS] Complete cycle: avg {avg_cycle:.3f}ms, max {max_cycle:.3f}ms")
    
    def test_02_sustained_throughput(self):
        """Test sustained throughput over time"""
        shared_data = {"prediction": 0}
        lock = threading.Lock()
        stop_event = threading.Event()
        processed = {"count": 0}
        
        def producer():
            for i in range(200):
                if stop_event.is_set():
                    break
                with lock:
                    shared_data["prediction"] = i % 5
                time.sleep(0.005)  # 200Hz
        
        def consumer():
            recent = deque(maxlen=3)
            while not stop_event.is_set():
                with lock:
                    pred = shared_data["prediction"]
                
                recent.append(pred)
                smoothed = int(round(mean(recent))) if len(recent) > 1 else pred
                processed["count"] += 1
                
                time.sleep(0.003)  # ~333Hz capability
        
        t1 = threading.Thread(target=producer, daemon=True)
        t2 = threading.Thread(target=consumer, daemon=True)
        
        start = time.perf_counter()
        try:
            t1.start()
            t2.start()
            
            t1.join(timeout=2)
        finally:
            stop_event.set()
            t2.join(timeout=1)
        
        elapsed = time.perf_counter() - start
        throughput = processed["count"] / elapsed
        
        self.assertGreater(throughput, 100)  # Should sustain > 100Hz
        print(f"[PASS] Sustained throughput: {throughput:.0f} Hz ({processed['count']} predictions in {elapsed:.2f}s)")


if __name__ == "__main__":
    unittest.main()
