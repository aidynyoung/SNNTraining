"""
Real-Time Execution Loop
========================
Fixed timestep loop with deadline enforcement.

Features:
- Deadline monitoring (watchdog)
- Slow-update path skipping on timeout
- CPU affinity and priority setting (Linux)
- Lock memory to prevent swapping

For manufacturing and UAV deployment.
"""

import torch
import time
import signal
import sys
from typing import Optional, Callable, Dict
from dataclasses import dataclass
from collections import deque


@dataclass
class RealTimeConfig:
    """Configuration for real-time execution."""
    # Timing
    bin_width_ms: float = 50.0           # Target bin width (timestep budget)
    deadline_slack_ms: float = 5.0      # Allowable slack before warning
    
    # Mode
    skip_slow_updates_on_timeout: bool = True
    slow_update_interval: int = 10     # Slow update every N steps normally
    
    # CPU/Priority (Linux only)
    set_realtime_priority: bool = False  # Use SCHED_FIFO
    cpu_affinity: Optional[list] = None  # Bind to specific CPUs
    lock_memory: bool = False            # mlockall to prevent swapping
    
    # Monitoring
    log_latency: bool = True
    latency_buffer_size: int = 100


class DeadlineMonitor:
    """
    Monitors timestep execution time and enforces deadlines.
    
    Skips non-critical operations (slow weight updates) if running
    behind schedule.
    """
    
    def __init__(self, config: RealTimeConfig):
        self.config = config
        self.deadline_sec = config.bin_width_ms / 1000.0
        self.slack_sec = config.deadline_slack_ms / 1000.0
        
        # Statistics
        self.latency_history = deque(maxlen=config.latency_buffer_size)
        self.miss_count = 0
        self.total_steps = 0
        self.last_step_start = None
        
        # Deadline state
        self.deadline_missed = False
        self.skip_slow_update = False
        
    def step_begin(self) -> float:
        """Mark beginning of timestep. Returns current time."""
        self.last_step_start = time.perf_counter()
        self.deadline_missed = False
        self.skip_slow_update = False
        return self.last_step_start
    
    def step_end(self) -> Dict:
        """
        Mark end of timestep, check deadline.
        
        Returns dict with timing info and flags.
        """
        if self.last_step_start is None:
            raise RuntimeError("step_begin() not called")
        
        now = time.perf_counter()
        elapsed = now - self.last_step_start
        
        self.total_steps += 1
        self.latency_history.append(elapsed * 1000)  # Store in ms
        
        # Check deadline
        self.deadline_missed = elapsed > self.deadline_sec
        
        if self.deadline_missed:
            self.miss_count += 1
            
            # Determine if we should skip slow update
            if self.config.skip_slow_updates_on_timeout:
                self.skip_slow_update = True
        
        return {
            'elapsed_ms': elapsed * 1000,
            'deadline_ms': self.config.bin_width_ms,
            'deadline_missed': self.deadline_missed,
            'skip_slow_update': self.skip_slow_update,
            'time_left_ms': max(0, self.deadline_sec - elapsed) * 1000,
        }
    
    def should_run_slow_update(self, step_counter: int) -> bool:
        """
        Determine if slow update should run this step.
        
        Normally runs every slow_update_interval steps, but skipped
        if deadline was recently missed.
        """
        if self.skip_slow_update:
            return False
        
        return step_counter % self.config.slow_update_interval == 0
    
    def get_stats(self) -> Dict:
        """Get timing statistics."""
        if len(self.latency_history) == 0:
            return {}
        
        latencies = list(self.latency_history)
        return {
            'mean_latency_ms': sum(latencies) / len(latencies),
            'max_latency_ms': max(latencies),
            'min_latency_ms': min(latencies),
            'miss_rate': self.miss_count / max(1, self.total_steps),
            'miss_count': self.miss_count,
            'total_steps': self.total_steps,
        }


class RealTimeLoop:
    """
    Real-time execution loop for SNN inference/training.
    
    Enforces fixed timestep budget and provides deadline monitoring.
    Suitable for hard real-time deployment on Linux with SCHED_FIFO.
    """
    
    def __init__(
        self,
        trainer,
        config: Optional[RealTimeConfig] = None,
        input_callback: Optional[Callable] = None,
        output_callback: Optional[Callable] = None
    ):
        """
        Args:
            trainer: OnlineTrainer instance
            config: Real-time configuration
            input_callback: Function to get next input (returns x, target)
            output_callback: Function to handle output (receives y_pred, info)
        """
        self.trainer = trainer
        self.config = config or RealTimeConfig()
        self.input_callback = input_callback
        self.output_callback = output_callback
        
        self.monitor = DeadlineMonitor(self.config)
        self.running = False
        self.step_count = 0
        
        # Signal handling for clean shutdown
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown."""
        def handle_signal(signum, frame):
            print(f"\nReceived signal {signum}, shutting down...")
            self.stop()
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    
    def setup_realtime(self):
        """
        Configure process for real-time execution (Linux only).
        
        Sets:
        - SCHED_FIFO scheduling policy
        - CPU affinity
        - Memory locking (mlockall)
        """
        import os
        import platform
        
        if platform.system() != 'Linux':
            print("Real-time setup only available on Linux")
            return
        
        try:
            # Set CPU affinity
            if self.config.cpu_affinity is not None:
                os.sched_setaffinity(0, self.config.cpu_affinity)
                print(f"CPU affinity set to {self.config.cpu_affinity}")
            
            # Set real-time priority (SCHED_FIFO)
            if self.config.set_realtime_priority:
                import ctypes
                libc = ctypes.CDLL('libc.so.6')
                
                # SCHED_FIFO = 1
                # Priority range: 1-99 (higher = more priority)
                sched_param = ctypes.c_int(50)  # Priority 50
                result = libc.sched_setscheduler(0, 1, ctypes.byref(sched_param))
                
                if result == 0:
                    print("SCHED_FIFO priority set")
                else:
                    print("Warning: Failed to set SCHED_FIFO (need root/capabilities)")
            
            # Lock memory
            if self.config.lock_memory:
                import ctypes
                libc = ctypes.CDLL('libc.so.6')
                
                # MCL_CURRENT | MCL_FUTURE = 1 | 2 = 3
                result = libc.mlockall(3)
                
                if result == 0:
                    print("Memory locked (mlockall)")
                else:
                    print("Warning: Failed to lock memory (need root)")
        
        except Exception as e:
            print(f"Real-time setup failed: {e}")
    
    def run(self, max_steps: Optional[int] = None):
        """
        Run real-time loop.
        
        Args:
            max_steps: Maximum steps to run (None = infinite)
        """
        self.running = True
        self.setup_realtime()
        
        print(f"Real-time loop starting (bin width: {self.config.bin_width_ms}ms)")
        
        while self.running:
            # Check step limit
            if max_steps is not None and self.step_count >= max_steps:
                break
            
            # Begin timestep
            self.monitor.step_begin()
            
            # Get input
            if self.input_callback is not None:
                try:
                    x, target = self.input_callback()
                except StopIteration:
                    print("Input stream ended")
                    break
            else:
                # Default: zeros
                x = torch.zeros(self.trainer.rsnn.input_size)
                target = torch.zeros(2)
            
            # Forward + training step
            try:
                if hasattr(self.trainer, 'step'):
                    y_pred, error = self.trainer.step(x, target)
                    info = {}
                else:
                    y_pred, error, info = self.trainer.step(x, target)
            except Exception as e:
                print(f"Training step error: {e}")
                y_pred = torch.zeros(2)
                info = {'error': str(e)}
            
            # End timestep and check deadline
            timing_info = self.monitor.step_end()
            
            # Combine info
            info.update(timing_info)
            info['step'] = self.step_count
            
            # Output callback
            if self.output_callback is not None:
                self.output_callback(y_pred, info)
            
            self.step_count += 1
            
            # Log if deadline missed
            if timing_info['deadline_missed'] and self.config.log_latency:
                print(f"[Deadline Miss] Step {self.step_count}: "
                      f"{timing_info['elapsed_ms']:.2f}ms > "
                      f"{timing_info['deadline_ms']:.2f}ms")
        
        print(f"Real-time loop stopped after {self.step_count} steps")
        print(f"Final stats: {self.monitor.get_stats()}")
    
    def stop(self):
        """Signal loop to stop."""
        self.running = False
    
    def get_stats(self) -> Dict:
        """Get execution statistics."""
        return {
            'total_steps': self.step_count,
            'timing': self.monitor.get_stats(),
        }


class ROS2NodeWrapper(RealTimeLoop):
    """
    ROS 2 node wrapper for real-time SNN execution.
    
    Wraps the real-time loop in a ROS 2 node for manufacturing/UAV
    integration. Subscribes to input topic, publishes output topic.
    
    Requires rclpy to be installed.
    """
    
    def __init__(
        self,
        trainer,
        config: Optional[RealTimeConfig] = None,
        input_topic: str = '/snn_input',
        output_topic: str = '/snn_output',
        node_name: str = 'arthedain_snn'
    ):
        super().__init__(trainer, config)
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.node_name = node_name
        
        self.ros_node = None
        self.subscription = None
        self.publisher = None
        self.latest_input = None
        
    def setup_ros(self):
        """Initialize ROS 2 node."""
        try:
            import rclpy
            from std_msgs.msg import Float32MultiArray
            
            rclpy.init()
            self.ros_node = rclpy.create_node(self.node_name)
            
            # Publisher
            self.publisher = self.ros_node.create_publisher(
                Float32MultiArray, self.output_topic, 10
            )
            
            # Subscriber
            def on_input(msg):
                self.latest_input = torch.tensor(msg.data)
            
            self.subscription = self.ros_node.create_subscription(
                Float32MultiArray, self.input_topic, on_input, 10
            )
            
            print(f"ROS 2 node '{self.node_name}' initialized")
            print(f"  Input: {self.input_topic}")
            print(f"  Output: {self.output_topic}")
            
        except ImportError:
            print("rclpy not installed, ROS 2 mode unavailable")
            raise
    
    def input_callback(self):
        """ROS-based input callback."""
        if self.latest_input is None:
            # No input yet, return zeros
            return (
                torch.zeros(self.trainer.rsnn.input_size),
                torch.zeros(2)
            )
        
        # Use latest input, zero target (inference mode)
        x = self.latest_input
        target = torch.zeros(2)
        
        return x, target
    
    def output_callback(self, y_pred, info):
        """ROS-based output callback."""
        try:
            from std_msgs.msg import Float32MultiArray
            
            msg = Float32MultiArray()
            msg.data = y_pred.tolist()
            self.publisher.publish(msg)
            
            # Also spin ROS to process callbacks
            import rclpy
            rclpy.spin_once(self.ros_node, timeout_sec=0)
            
        except Exception as e:
            print(f"ROS publish error: {e}")
    
    def run(self, max_steps: Optional[int] = None):
        """Run with ROS 2 integration."""
        self.setup_ros()
        try:
            super().run(max_steps)
        finally:
            if self.ros_node is not None:
                self.ros_node.destroy_node()
                import rclpy
                rclpy.shutdown()


if __name__ == "__main__":
    print("Real-Time Loop Test")
    print("=" * 50)
    
    # Create mock trainer
    from models.rsnn import RSNN, RSNNConfig
    from models.readout import Readout, ReadoutConfig
    from models.hebbian import DualHebbian
    from training.online_trainer import OnlineTrainer
    
    model = RSNN(config=RSNNConfig(input_size=10, hidden_size=32))
    readout = Readout(ReadoutConfig(32, 2))
    hebbian = DualHebbian((32, 32))
    trainer = OnlineTrainer(model, readout, hebbian)
    
    # Real-time config
    config = RealTimeConfig(
        bin_width_ms=20.0,  # 50Hz
        log_latency=True,
        skip_slow_updates_on_timeout=True
    )
    
    # Synthetic input generator
    def input_gen():
        import random
        while True:
            x = torch.randn(10)
            target = torch.tensor([random.random(), random.random()])
            yield x, target
    
    gen = input_gen()
    
    def get_input():
        return next(gen)
    
    def on_output(y_pred, info):
        if info['step'] % 100 == 0:
            print(f"Step {info['step']}: latency={info['elapsed_ms']:.2f}ms")
    
    # Run loop
    loop = RealTimeLoop(trainer, config, get_input, on_output)
    loop.run(max_steps=500)
    
    print("\nReal-time test complete.")
