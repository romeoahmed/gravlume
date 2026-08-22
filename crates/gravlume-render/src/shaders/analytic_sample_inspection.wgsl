fn inspected_scene_value(sample: GeometricSample) -> vec4<f32> {
    return scene_value(sample.termination, sample.source_coordinates);
}
