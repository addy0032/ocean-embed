import React, { useRef, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Text, Line } from '@react-three/drei';
import * as THREE from 'three';
import { gridToCanvas } from '../utils/colormap';
import { RotateCcw } from 'lucide-react';

interface OceanSceneProps {
  depths: number[];
  selectedDepth: number;
  onSelectDepth: (depth: number) => void;
  activeValues: (number | null)[][];
  minVal: number;
  maxVal: number;
  isError?: boolean;
  verticalExaggeration: number; // 1 to 10
}

// Plane dimensions in 3D scene units
const SCENE_WIDTH = 24.1;  // Longitude X: 45° to 105° (60 deg span)
const SCENE_DEPTH = 10.1;  // Latitude Z: 5° to 30° (25 deg span)
const MAX_VERTICAL_SPAN = 8.0; // Total height in scene units at 1x exaggeration

// Individual depth plane mesh
const DepthPlane: React.FC<{
  depth: number;
  index: number;
  totalDepths: number;
  yPos: number;
  isSelected: boolean;
  texture: THREE.CanvasTexture | null;
  onSelect: () => void;
}> = ({ depth, yPos, isSelected, texture, onSelect }) => {
  return (
    <group position={[0, yPos, 0]}>
      {/* Plane surface */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        onClick={(e) => {
          e.stopPropagation();
          onSelect();
        }}
      >
        <planeGeometry args={[SCENE_WIDTH, SCENE_DEPTH]} />
        <meshBasicMaterial
          map={texture || undefined}
          color={texture ? '#ffffff' : (isSelected ? '#0ea5e9' : '#334155')}
          transparent
          opacity={isSelected ? 0.95 : 0.28}
          side={THREE.DoubleSide}
          depthWrite={isSelected}
        />
      </mesh>

      {/* Plane outline border */}
      <lineSegments>
        <edgesGeometry args={[new THREE.PlaneGeometry(SCENE_WIDTH, SCENE_DEPTH)]} />
        <lineBasicMaterial
          color={isSelected ? '#38bdf8' : '#475569'}
          linewidth={isSelected ? 3 : 1}
          transparent
          opacity={isSelected ? 1.0 : 0.4}
        />
      </lineSegments>

      {/* Selected highlight glow boundary */}
      {isSelected && (
        <group position={[0, 0, 0]}>
          <lineSegments>
            <edgesGeometry args={[new THREE.BoxGeometry(SCENE_WIDTH + 0.1, 0.05, SCENE_DEPTH + 0.1)]} />
            <lineBasicMaterial color="#38bdf8" linewidth={2} />
          </lineSegments>
          {/* Depth tag floating next to active plane */}
          <Text
            position={[SCENE_WIDTH / 2 + 1.2, 0, 0]}
            fontSize={0.6}
            color="#38bdf8"
            anchorX="left"
            anchorY="middle"
          >
            {`${depth} m`}
          </Text>
        </group>
      )}
    </group>
  );
};

// Vertical depth indicator axis on the side
const DepthAxisIndicator: React.FC<{
  depths: number[];
  depthPositions: number[];
  selectedDepth: number;
  selectedY: number;
  onSelectDepth: (depth: number) => void;
}> = ({ depths, depthPositions, selectedDepth, selectedY, onSelectDepth }) => {
  const axisX = -SCENE_WIDTH / 2 - 1.2;
  const axisZ = -SCENE_DEPTH / 2;
  const minY = depthPositions[depthPositions.length - 1];
  const maxY = depthPositions[0];

  const axisLinePoints = useMemo(() => [
    new THREE.Vector3(axisX, maxY + 0.5, axisZ),
    new THREE.Vector3(axisX, minY - 0.5, axisZ),
  ], [axisX, maxY, minY, axisZ]);

  return (
    <group>
      {/* Vertical axis line */}
      <Line
        points={axisLinePoints}
        color="#64748b"
        lineWidth={2}
      />

      {/* Depth tick marks & labels */}
      {depths.map((d, i) => {
        const y = depthPositions[i];
        const isSelected = d === selectedDepth;
        return (
          <group key={d} position={[axisX, y, axisZ]}>
            {/* Tick mark */}
            <Line
              points={[
                new THREE.Vector3(0, 0, 0),
                new THREE.Vector3(0.5, 0, 0),
              ]}
              color={isSelected ? '#38bdf8' : '#64748b'}
              lineWidth={isSelected ? 3 : 1}
            />
            {/* Label */}
            <Text
              position={[-0.4, 0, 0]}
              fontSize={0.35}
              color={isSelected ? '#38bdf8' : '#94a3b8'}
              anchorX="right"
              anchorY="middle"
              onClick={(e) => {
                e.stopPropagation();
                onSelectDepth(d);
              }}
            >
              {`${d}m`}
            </Text>
          </group>
        );
      })}

      {/* Active depth marker sphere */}
      <mesh position={[axisX, selectedY, axisZ]}>
        <sphereGeometry args={[0.25, 16, 16]} />
        <meshBasicMaterial color="#38bdf8" />
      </mesh>
    </group>
  );
};

// Domain Bounding Box and Geographic Axis Labels
const DomainBoundingBox: React.FC<{
  maxY: number;
  minY: number;
}> = ({ maxY, minY }) => {
  const height = Math.abs(maxY - minY) || 1;
  const centerY = (maxY + minY) / 2;

  return (
    <group>
      {/* Box wireframe */}
      <lineSegments position={[0, centerY, 0]}>
        <edgesGeometry args={[new THREE.BoxGeometry(SCENE_WIDTH, height, SCENE_DEPTH)]} />
        <lineBasicMaterial color="#334155" transparent opacity={0.35} />
      </lineSegments>

      {/* Longitude Labels (X Axis) */}
      <Text position={[-SCENE_WIDTH / 2, maxY + 0.6, SCENE_DEPTH / 2 + 0.6]} fontSize={0.45} color="#94a3b8">
        45°E
      </Text>
      <Text position={[0, maxY + 0.6, SCENE_DEPTH / 2 + 0.6]} fontSize={0.45} color="#94a3b8">
        75°E (Longitude →)
      </Text>
      <Text position={[SCENE_WIDTH / 2, maxY + 0.6, SCENE_DEPTH / 2 + 0.6]} fontSize={0.45} color="#94a3b8">
        105°E
      </Text>

      {/* Latitude Labels (Z Axis) */}
      <Text position={[SCENE_WIDTH / 2 + 0.8, maxY + 0.6, SCENE_DEPTH / 2]} fontSize={0.45} color="#94a3b8">
        5°N
      </Text>
      <Text position={[SCENE_WIDTH / 2 + 0.8, maxY + 0.6, 0]} fontSize={0.45} color="#94a3b8">
        17.5°N
      </Text>
      <Text position={[SCENE_WIDTH / 2 + 0.8, maxY + 0.6, -SCENE_DEPTH / 2]} fontSize={0.45} color="#94a3b8">
        30°N (Latitude ↑)
      </Text>
    </group>
  );
};

// Controller to enable "Reset View"
const CameraManager: React.FC<{
  controlsRef: React.MutableRefObject<any>;
}> = ({ controlsRef }) => {
  return (
    <OrbitControls
      ref={controlsRef}
      enableDamping
      dampingFactor={0.08}
      minDistance={5}
      maxDistance={60}
      target={[0, -2, 0]}
    />
  );
};

export const OceanScene: React.FC<OceanSceneProps> = ({
  depths,
  selectedDepth,
  onSelectDepth,
  activeValues,
  minVal,
  maxVal,
  isError = false,
  verticalExaggeration = 3.0,
}) => {
  const controlsRef = useRef<any>(null);

  // Generate Three.js CanvasTexture for the active layer
  const activeTexture = useMemo(() => {
    if (!activeValues || activeValues.length === 0) return null;
    const canvas = gridToCanvas(activeValues, minVal, maxVal, isError);
    if (!canvas) return null;
    const tex = new THREE.CanvasTexture(canvas);
    tex.magFilter = THREE.NearestFilter;
    tex.minFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    return tex;
  }, [activeValues, minVal, maxVal, isError]);

  // Compute depth plane positions along Y axis (downward, negative Y)
  // Non-linear depth distribution with vertical exaggeration
  const depthPositions = useMemo(() => {
    if (!depths || depths.length === 0) return [];
    const maxDepth = depths[depths.length - 1] || 1000;

    return depths.map((d) => {
      // Use square root scaling so shallow thermocline levels (0-150m) are well resolved
      const normDepth = Math.sqrt(d / maxDepth);
      const y = -normDepth * (MAX_VERTICAL_SPAN * (verticalExaggeration / 3.0));
      return y;
    });
  }, [depths, verticalExaggeration]);

  const selectedIdx = depths.indexOf(selectedDepth);
  const selectedY = selectedIdx >= 0 ? depthPositions[selectedIdx] : 0;
  const maxY = depthPositions[0] ?? 0;
  const minY = depthPositions[depthPositions.length - 1] ?? -MAX_VERTICAL_SPAN;

  const handleResetView = () => {
    if (controlsRef.current) {
      controlsRef.current.reset();
    }
  };

  return (
    <div className="relative w-full h-[520px] bg-slate-950 rounded-xl overflow-hidden border border-slate-800 shadow-2xl">
      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [20, 16, 26], fov: 42 }}
        className="w-full h-full"
      >
        <color attach="background" args={['#090d16']} />
        <ambientLight intensity={1.2} />
        <directionalLight position={[10, 20, 15]} intensity={1.0} />

        <CameraManager controlsRef={controlsRef} />

        {/* 15 Depth Layer Planes */}
        {depths.map((d, i) => {
          const isSelected = d === selectedDepth;
          return (
            <DepthPlane
              key={d}
              depth={d}
              index={i}
              totalDepths={depths.length}
              yPos={depthPositions[i]}
              isSelected={isSelected}
              texture={isSelected ? activeTexture : null}
              onSelect={() => onSelectDepth(d)}
            />
          );
        })}

        {/* Vertical Depth Indicator Axis */}
        <DepthAxisIndicator
          depths={depths}
          depthPositions={depthPositions}
          selectedDepth={selectedDepth}
          selectedY={selectedY}
          onSelectDepth={onSelectDepth}
        />

        {/* Bounding Box & Geographic Labels */}
        <DomainBoundingBox maxY={maxY} minY={minY} />
      </Canvas>

      {/* Floating Controls Overlay */}
      <div className="absolute top-4 right-4 flex items-center gap-2 bg-slate-900/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-800 shadow-lg text-xs text-slate-300 z-10">
        <button
          onClick={handleResetView}
          className="flex items-center gap-1.5 px-2.5 py-1 bg-slate-800 hover:bg-slate-700 active:scale-95 text-slate-200 rounded transition border border-slate-700"
          title="Reset Camera Orientation"
        >
          <RotateCcw className="w-3.5 h-3.5 text-cyan-400" />
          <span>Reset View</span>
        </button>
      </div>

      {/* Legend / Orientation Helper Badge */}
      <div className="absolute bottom-4 left-4 bg-slate-900/85 backdrop-blur-md px-3 py-2 rounded-lg border border-slate-800 text-[11px] font-mono text-slate-400 pointer-events-none">
        <div className="flex items-center gap-2 text-slate-200 font-semibold mb-1">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span>3D Multi-Layer Subsurface View</span>
        </div>
        <div>X: Longitude (45°E → 105°E)</div>
        <div>Z: Latitude (5°N → 30°N)</div>
        <div>Y: Depth (0m → 1000m downward)</div>
      </div>
    </div>
  );
};
