# functional_twin.py
import numpy as np
import time
import json
import asyncio
import websockets
from scipy.integrate import odeint

# === PHYSICS PARAMETERS (ARDIDEN 1H1 SPECIFIC) ===
# From TCDS E.037 and derived values
TANK_CAPACITY = 1400  # liters
FUEL_DENSITY = 0.81   # kg/L (Jet A-1)
MAX_FLOW_RATE = 279   # L/hr (30-sec OEI)
HP_PRESSURE_RANGE = (5500, 8300)  # kPa
N1_MAX = 40095        # rpm (Take-off)

# === CORE COMPONENT MODELS ===
class FuelTank:
    def __init__(self):
        self.volume = TANK_CAPACITY
        self.temperature = 15  # °C
        self._crossfeed_valve = False
        
    def update(self, flow_out, dt, return_flow=0):
        # Density-temperature relationship: ρ = ρ₀[1 - α(T - T₀)]
        density = FUEL_DENSITY * (1 - 7e-4 * (self.temperature - 15))
        consumed = flow_out * dt / 3600  # Convert L/hr to L/s
        self.volume = max(0, self.volume - consumed + return_flow)
        return density

class BoostPump:
    def __init__(self):
        self.pressure = 300  # kPa
        self.is_active = True
        self.flow_rate = 0
        
    def update(self, demand_flow, n1_rpm):
        # Pump characteristic curve
        if not self.is_active:
            return 0
            
        self.flow_rate = min(500, demand_flow * 1.2)  # 20% safety margin
        self.pressure = 300 + 100 * (self.flow_rate/500)**2
        return self.flow_rate

class FuelFilter:
    def __init__(self):
        self.contamination = 0  # 0-1 scale
        self.delta_p = 5  # kPa (clean)
        
    def update(self, flow_rate, dt):
        # Contamination increases with operational time
        self.contamination += 0.00001 * dt
        self.delta_p = 5 + 40 * self.contamination  # Max 45 kPa at contamination=1
        return self.delta_p

class FADEC:
    def __init__(self):
        self.channel_A = True
        self.channel_B = True
        self.control_freq = 70  # Hz
        self.last_update = time.time()
        
    def calculate_fuel_flow(self, throttle, n1, n2, t45, altitude):
        # Power mapping (throttle % to kW)
        power_map = {0:0, 25:300, 50:450, 75:568, 100:640}
        p_demand = power_map.get(throttle, 0)
        
        # Fuel flow calculation (kg/hr)
        thermal_eff = 0.30
        calorific_value = 43000  # kJ/kg
        fuel_flow_kg = (p_demand / (thermal_eff * calorific_value)) * 3600
        
        # Limit protection logic
        if t45 > 1201:  # Take-off temp limit
            fuel_flow_kg *= 0.98
        if n1 > N1_MAX:
            fuel_flow_kg *= 0.95
            
        return fuel_flow_kg / FUEL_DENSITY  # Convert to L/hr

class MeteringValve:
    def __init__(self):
        self.position = 0  # 0-100%
        self.response_time = 0.1  # seconds
        
    def update(self, command, dt):
        # First-order response to command
        self.position += (command - self.position) * (dt / self.response_time)
        return self.position

class HealthMonitor:
    def __init__(self):
        self.components = {
            'filter': 100,
            'hp_pump': 100,
            'injectors': 100
        }
        
    def update(self, system_state):
        # Filter health based on pressure drop
        dp = system_state['filter_dp']
        filter_health = 100 - 30 * min(1, max(0, (dp - 10)/30))
        
        # Injector health based on flow deviation
        flow_dev = abs(system_state['flow_cmd'] - system_state['flow_act'])/system_state['flow_cmd']
        injector_health = 100 - 25 * min(1, flow_dev/0.1)
        
        # Composite health
        self.components = {
            'filter': max(0, filter_health),
            'injectors': max(0, injector_health),
            'overall': 0.4*filter_health + 0.6*injector_health
        }
        return self.components

# === MAIN SIMULATION ENGINE ===
class FuelSystemSimulator:
    def __init__(self):
        self.components = {
            'tank': FuelTank(),
            'boost_pump': BoostPump(),
            'filter': FuelFilter(),
            'fadec': FADEC(),
            'metering_valve': MeteringValve(),
            'health': HealthMonitor()
        }
        self.state = {
            'throttle': 0,
            'n1': 0,
            'n2': 0,
            't45': 0,
            'altitude': 0,
            'fuel_quantity': TANK_CAPACITY,
            'flow_cmd': 0,
            'flow_act': 0,
            'hp_pressure': 0,
            'filter_dp': 0,
            'valve_position': 0,
            'health': {}
        }
        self.time = 0
        self.faults = {}
        
    def apply_fault(self, sensor, fault_type):
        self.faults[sensor] = fault_type
        
    def get_sensor_reading(self, true_value, sensor_name):
        fault_type = self.faults.get(sensor_name, None)
        if not fault_type:
            return true_value
            
        if fault_type == "STUCK":
            return 500  # Fixed value
        elif fault_type == "DRIFT":
            return true_value * (0.95 - 0.001*self.time)  # 5%/min drift
        elif fault_type == "NOISE":
            return true_value * (1 + 0.02*np.random.normal())
        return true_value
        
    def update(self, dt):
        # Update engine parameters based on throttle
        self.state['n1'] = 39000 * (self.state['throttle']/100)**0.8
        self.state['t45'] = 1073 + 130 * (self.state['throttle']/100)
        
        # FADEC fuel calculation
        flow_cmd = self.components['fadec'].calculate_fuel_flow(
            self.state['throttle'],
            self.state['n1'],
            self.state['n2'],
            self.state['t45'],
            self.state['altitude']
        )
        self.state['flow_cmd'] = flow_cmd
        
        # Valve control
        valve_cmd = min(100, max(0, (flow_cmd/MAX_FLOW_RATE)*100))
        valve_pos = self.components['metering_valve'].update(valve_cmd, dt)
        self.state['valve_position'] = valve_pos
        
        # Actual flow based on valve position
        actual_flow = (valve_pos/100) * MAX_FLOW_RATE
        self.state['flow_act'] = actual_flow
        
        # Update tank
        density = self.components['tank'].update(actual_flow, dt)
        self.state['fuel_quantity'] = self.components['tank'].volume
        
        # Update filter
        dp = self.components['filter'].update(actual_flow, dt)
        self.state['filter_dp'] = dp
        
        # Update HP pressure
        self.state['hp_pressure'] = HP_PRESSURE_RANGE[0] + (
            (HP_PRESSURE_RANGE[1]-HP_PRESSURE_RANGE[0]) * (valve_pos/100)
        )
        
        # Health assessment
        health = self.components['health'].update(self.state)
        self.state['health'] = health
        
        self.time += dt
        
        # Apply sensor faults
        self.state['fuel_quantity_display'] = self.get_sensor_reading(
            self.state['fuel_quantity'], 'fuel_probe'
        )
        
        return self.state

# === WEBSOCKET SERVER ===
async def simulation_server(websocket, path):
    simulator = FuelSystemSimulator()
    last_time = time.time()
    
    while True:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        state = simulator.update(dt)
        await websocket.send(json.dumps(state))
        await asyncio.sleep(0.01)  # 100 Hz update rate

if __name__ == "__main__":
    start_server = websockets.serve(simulation_server, "0.0.0.0", 8765)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()