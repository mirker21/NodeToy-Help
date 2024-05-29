import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import Plank from './3D_components/Plank';
import HandcarTopHalf from './3D_components/HandcarTopHalf';

export default function ProjectCanvas() {
    return (
        <Canvas id="project-canvas">
            <OrbitControls/>
            <ambientLight intensity={5} />
            <color args={ [ '#11eeFF' ] } attach="background" />
            <pointLight position={[10, 10, 10]} />
            {/* <mesh>
                <boxGeometry />
                <meshStandardMaterial color="hotpink" />
            </mesh> */}
            <HandcarTopHalf/>
            <Plank/>
        </Canvas>
    )
}