// Test-only ordered sample corpus. Runtime-sized storage arrays keep the allocation proportional
// to requested evidence rather than viewport area. Each invocation owns one request and record;
// no inter-invocation synchronization or cross-workgroup visibility is required.
// Sources:
// - https://www.w3.org/TR/WGSL/#buffer-binding-determines-runtime-sized-array-element-count
// - https://www.w3.org/TR/WGSL/#compute-shader-workgroups

@group(0) @binding(10)
var<storage, read> corpus_requests: array<SampleInspectionRequest>;

@group(0) @binding(11)
var<storage, read_write> corpus_records: array<SampleInspectionRecord>;

@compute @workgroup_size(TRACE_WORKGROUP_AXIS, TRACE_WORKGROUP_AXIS, 1)
fn inspect_sample_corpus(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let invocations_per_workgroup = TRACE_WORKGROUP_AXIS * TRACE_WORKGROUP_AXIS;
    let index = workgroup_id.x * invocations_per_workgroup + local_index;
    if index >= arrayLength(&corpus_requests) {
        return;
    }

    let request = corpus_requests[index];
    let sample = trace_pixel_at(
        request.pixel_extent.xy,
        request.pixel_extent.zw,
        request.subpixel.xy,
    );
    corpus_records[index] = encode_inspected_sample(sample, inspected_scene_value(sample));
}
