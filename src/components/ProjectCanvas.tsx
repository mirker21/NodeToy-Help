import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import HandcarTopHalf from './3D_components/HandcarTopHalf';

export default function ProjectCanvas() {
    return (
        <Canvas id="project-canvas">
            <OrbitControls/>
            <ambientLight intensity={5} />
            <color args={ [ '#11eeFF' ] } attach="background" />
            <pointLight position={[10, 10, 10]} />
            <HandcarTopHalf/>
        </Canvas>
    )
}
