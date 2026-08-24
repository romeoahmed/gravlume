use super::super::{TRACE_WORKGROUP_AXIS, TracePipeline, TracePlan, shader};

const INVOCATIONS_PER_WORKGROUP: u32 = TRACE_WORKGROUP_AXIS * TRACE_WORKGROUP_AXIS;

pub(super) const fn inspection_workgroup_count(sample_count: u32) -> u32 {
    sample_count.div_ceil(INVOCATIONS_PER_WORKGROUP)
}

pub(super) struct SampleInspectionKernel {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl SampleInspectionKernel {
    pub(super) fn new(
        device: &wgpu::Device,
        trace: &TracePipeline,
        requests: &wgpu::Buffer,
        records: &wgpu::Buffer,
    ) -> Self {
        let shader_source = match trace.plan {
            TracePlan::AcceleratedSky => shader::analytic_sample_inspection(),
            TracePlan::EquatorialBolometricSurface => shader::bolometric_sample_inspection(),
            TracePlan::EquatorialBlackbodySurface => shader::blackbody_sample_inspection(),
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sample inspection shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        let pipeline_constants = [(
            "SURFACE_EVENTS_ENABLED",
            trace.plan.surface_events_enabled(),
        )];
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("inspect_samples"),
            // This kernel owns its sole pipeline and bind group. Deriving their private layout
            // keeps shader resource declarations authoritative without creating a shared seam.
            // Source: https://docs.rs/wgpu/30.0.1/wgpu/struct.ComputePipelineDescriptor.html#structfield.layout
            layout: None,
            module: &shader,
            entry_point: Some("inspect_samples"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &pipeline_constants,
                ..Default::default()
            },
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: trace.uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: records.as_entire_binding(),
            },
        ];
        if let Some(blackbody_lut) = &trace.blackbody_lut {
            entries.push(wgpu::BindGroupEntry {
                binding: 8,
                resource: blackbody_lut.as_entire_binding(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 9,
            resource: requests.as_entire_binding(),
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sample inspection bind group"),
            layout: &layout,
            entries: &entries,
        });
        Self {
            pipeline,
            bind_group,
        }
    }

    pub(super) fn encode_samples(&self, encoder: &mut wgpu::CommandEncoder, sample_count: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sample inspection pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(inspection_workgroup_count(sample_count), 1, 1);
    }
}
