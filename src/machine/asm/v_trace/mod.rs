pub mod infer;
mod utils;

use crate::{
    decoder::build_decoder,
    instructions::{
        blank_instruction, common, execute, execute_instruction, extract_opcode,
        generate_handle_function_list, instruction_length, is_basic_block_end_instruction,
        is_slowpath_instruction, v_alu as alu, HandleFunction, Instruction, Itype, Register,
        VVtype, VXtype,
    },
    machine::{
        asm::{
            check_memory, check_memory_inited, check_permission, ckb_vm_asm_labels,
            ckb_vm_x64_execute, v_trace::infer::VInferMachine, AotCode, AsmCoreMachine, Error,
        },
        VERSION0,
    },
    memory::{get_page_indices, FLAG_DIRTY, FLAG_WRITABLE},
    CoreMachine, DefaultMachine, Machine, Memory, SupportMachine,
};
use bytes::Bytes;
use ckb_vm_definitions::{
    asm::{
        calculate_slot, Trace, RET_CYCLES_OVERFLOW, RET_DECODE_TRACE, RET_DYNAMIC_JUMP, RET_EBREAK,
        RET_ECALL, RET_INVALID_PERMISSION, RET_MAX_CYCLES_EXCEEDED, RET_OUT_OF_BOUND,
        RET_SLOWPATH_TRACE, TRACE_ITEM_LENGTH,
    },
    instructions::{self as insts, OP_CUSTOM_TRACE_END},
    ISA_MOP, VLEN,
};
use eint::{Eint, E256, E8};
use lru::LruCache;
use std::mem::{transmute, MaybeUninit};

pub const VTRACE_MAX_LENGTH: usize = 32;

// This simply wraps AsmCoreMachine so we can introduce custom V processing functions
pub struct VTraceCoreMachine {
    inner: Box<AsmCoreMachine>,
}

impl VTraceCoreMachine {
    fn v_ref(&self, reg: usize, sew: u64, skip: usize, count: usize) -> &[u8] {
        let lb = (sew as usize) >> 3;
        let i0 = reg * (VLEN >> 3) + lb * skip;
        let len = lb * count;
        let i1 = i0 + len;

        &self.inner.register_file[i0..i1]
    }

    fn v_mut(&mut self, reg: usize, sew: u64, skip: usize, count: usize) -> &mut [u8] {
        let lb = (sew as usize) >> 3;
        let i0 = reg * (VLEN >> 3) + lb * skip;
        let len = lb * count;
        let i1 = i0 + len;

        &mut self.inner.register_file[i0..i1]
    }

    // Shortcuts(think DMA) to transfer data between vector registers and memories.
    // On the one hand, it eliminates all the costy Memory operations as well as `to_vec`
    // to work around life times.
    //
    // Another interesting use case here, is that symbolic executions can be leveraged
    // to eliminate loads. For instance, when executing the following snippet:
    // * vle256.v v0, (t3)
    // * vle256.v v4, (t4)
    // * vmul.vv v0, v0, v4
    // There is no need to do the first load into v0, instead, one can redirect one
    // operand of vmul.vv to load from memory directly.
    //
    // TODO: those are just an initial attempt for a PoC on rvv optimiztaions, it really
    // is necessary to revise all the V related trait methods.
    fn v_to_mem(
        &mut self,
        reg: usize,
        sew: u64,
        skip: usize,
        count: usize,
        addr: u64,
    ) -> Result<(), Error> {
        let lb = (sew as usize) >> 3;
        let i0 = reg * (VLEN >> 3) + lb * skip;
        let len = lb * count;
        let i1 = i0 + len;

        let page_indices = get_page_indices(addr, len as u64)?;
        for page in page_indices.0..=page_indices.1 {
            check_permission(&mut self.inner, page, FLAG_WRITABLE)?;
            check_memory(&mut self.inner, page);
            self.inner.set_flag(page, FLAG_DIRTY)?;
        }

        self.inner.memory[addr as usize..addr as usize + len]
            .copy_from_slice(&self.inner.register_file[i0..i1]);

        Ok(())
    }

    fn mem_to_v(
        &mut self,
        reg: usize,
        sew: u64,
        skip: usize,
        count: usize,
        addr: u64,
    ) -> Result<(), Error> {
        let lb = (sew as usize) >> 3;
        let i0 = reg * (VLEN >> 3) + lb * skip;
        let len = lb * count;
        let i1 = i0 + len;

        check_memory_inited(&mut self.inner, addr, len as usize)?;

        self.inner.register_file[i0..i1]
            .copy_from_slice(&self.inner.memory[addr as usize..addr as usize + len]);

        Ok(())
    }

    fn v_to_v(
        &mut self,
        reg: usize,
        sew: u64,
        skip: usize,
        count: usize,
        target_reg: usize,
    ) -> Result<(), Error> {
        let lb = (sew as usize) >> 3;
        let len = lb * count;

        let i0 = reg * (VLEN >> 3) + lb * skip;
        let i1 = i0 + len;

        let j0 = target_reg * (VLEN >> 3) + lb * skip;

        self.inner.register_file.copy_within(i0..i1, j0);

        Ok(())
    }
}

impl From<Box<AsmCoreMachine>> for VTraceCoreMachine {
    fn from(m: Box<AsmCoreMachine>) -> VTraceCoreMachine {
        VTraceCoreMachine { inner: m }
    }
}

impl CoreMachine for VTraceCoreMachine {
    type REG = u64;
    type MEM = <Box<AsmCoreMachine> as CoreMachine>::MEM;

    fn pc(&self) -> &Self::REG {
        self.inner.pc()
    }

    fn update_pc(&mut self, pc: Self::REG) {
        self.inner.update_pc(pc)
    }

    fn commit_pc(&mut self) {
        self.inner.commit_pc()
    }

    fn memory(&self) -> &Self::MEM {
        self.inner.memory()
    }

    fn memory_mut(&mut self) -> &mut Self::MEM {
        self.inner.memory_mut()
    }

    fn registers(&self) -> &[Self::REG] {
        self.inner.registers()
    }

    fn set_register(&mut self, idx: usize, value: Self::REG) {
        self.inner.set_register(idx, value)
    }

    fn isa(&self) -> u8 {
        self.inner.isa()
    }

    fn version(&self) -> u32 {
        self.inner.version()
    }

    fn element_ref(&self, reg: usize, sew: u64, n: usize) -> &[u8] {
        self.inner.element_ref(reg, sew, n)
    }

    fn element_mut(&mut self, reg: usize, sew: u64, n: usize) -> &mut [u8] {
        self.inner.element_mut(reg, sew, n)
    }

    fn get_bit(&self, reg: usize, n: usize) -> bool {
        self.inner.get_bit(reg, n)
    }

    fn set_bit(&mut self, reg: usize, n: usize) {
        self.inner.set_bit(reg, n)
    }

    fn clr_bit(&mut self, reg: usize, n: usize) {
        self.inner.clr_bit(reg, n)
    }

    fn set_vl(&mut self, rd: usize, rs1: usize, avl: u64, new_type: u64) {
        self.inner.set_vl(rd, rs1, avl, new_type)
    }

    fn vl(&self) -> u64 {
        self.inner.vl()
    }

    fn vlmax(&self) -> u64 {
        self.inner.vlmax()
    }

    fn vsew(&self) -> u64 {
        self.inner.vsew()
    }

    fn vlmul(&self) -> f64 {
        self.inner.vlmul()
    }

    fn vta(&self) -> bool {
        self.inner.vta()
    }

    fn vma(&self) -> bool {
        self.inner.vma()
    }

    fn vill(&self) -> bool {
        self.inner.vill()
    }

    fn vlenb(&self) -> u64 {
        self.inner.vlenb()
    }
}

impl SupportMachine for VTraceCoreMachine {
    fn cycles(&self) -> u64 {
        self.inner.cycles()
    }

    fn set_cycles(&mut self, cycles: u64) {
        self.inner.set_cycles(cycles)
    }

    fn max_cycles(&self) -> u64 {
        self.inner.max_cycles()
    }

    fn reset(&mut self, max_cycles: u64) {
        self.inner.reset(max_cycles)
    }

    fn reset_signal(&mut self) -> bool {
        self.inner.reset_signal()
    }

    fn running(&self) -> bool {
        self.inner.running()
    }

    fn set_running(&mut self, running: bool) {
        self.inner.set_running(running)
    }

    #[cfg(feature = "pprof")]
    fn code(&self) -> &Bytes {
        self.inner.code()
    }
}

type CM<'a> = DefaultMachine<'a, VTraceCoreMachine>;

pub struct VTrace<'a> {
    pub address: u64,
    pub code_length: u8,
    pub last_inst_length: u8,
    pub actions: Vec<Box<dyn Fn(&mut CM<'a>) -> Result<(), Error>>>,
}

impl<'a> Default for VTrace<'a> {
    fn default() -> Self {
        VTrace {
            address: 0,
            code_length: 0,
            last_inst_length: 0,
            actions: Vec::new(),
        }
    }
}

pub struct VTraceAsmMachine<'a> {
    pub machine: CM<'a>,
    pub aot_code: Option<&'a AotCode>,
    pub v_traces: LruCache<u64, VTrace<'a>>,
}

impl<'a> VTraceAsmMachine<'a> {
    pub fn new(machine: CM<'a>, aot_code: Option<&'a AotCode>) -> Self {
        let mut r = Self {
            machine,
            aot_code,
            v_traces: LruCache::new(128),
        };
        // Default to illegal configuration
        r.machine.set_vl(0, 0, 0, u64::MAX);
        r
    }

    pub fn set_max_cycles(&mut self, cycles: u64) {
        self.machine.inner.inner.max_cycles = cycles;
    }

    pub fn load_program(&mut self, program: &Bytes, args: &[Bytes]) -> Result<u64, Error> {
        self.machine.load_program(program, args)
    }

    pub fn run(&mut self) -> Result<i8, Error> {
        if self.machine.isa() & ISA_MOP != 0 && self.machine.version() == VERSION0 {
            return Err(Error::InvalidVersion);
        }
        let mut decoder = build_decoder::<u64>(self.machine.isa(), self.machine.version());
        let handle_function_list = generate_handle_function_list::<VInferMachine>();
        let handle_function_list_2 = generate_handle_function_list::<CM<'a>>();
        self.machine.set_running(true);
        while self.machine.running() {
            if self.machine.reset_signal() {
                decoder.reset_instructions_cache();
                self.aot_code = None;
            }
            let result = if let Some(aot_code) = &self.aot_code {
                if let Some(offset) = aot_code.labels.get(self.machine.pc()) {
                    let base_address = aot_code.base_address();
                    let offset_address = base_address + u64::from(*offset);
                    let f = unsafe {
                        transmute::<u64, fn(*mut AsmCoreMachine, u64) -> u8>(base_address)
                    };
                    f(&mut (*self.machine.inner_mut().inner), offset_address)
                } else {
                    unsafe { ckb_vm_x64_execute(&mut (*self.machine.inner_mut().inner)) }
                }
            } else {
                unsafe { ckb_vm_x64_execute(&mut (*self.machine.inner_mut().inner)) }
            };
            match result {
                RET_DECODE_TRACE => {
                    let pc = *self.machine.pc();
                    let slot = calculate_slot(pc);
                    let mut trace = Trace::default();
                    let mut current_pc = pc;
                    let mut i = 0;
                    while i < TRACE_ITEM_LENGTH {
                        let instruction = decoder.decode(self.machine.memory_mut(), current_pc)?;
                        let end_instruction = is_basic_block_end_instruction(instruction);
                        let length = instruction_length(instruction);
                        trace.last_inst_length = length;
                        current_pc += u64::from(length);
                        if trace.slowpath == 0 && is_slowpath_instruction(instruction) {
                            trace.slowpath = 1;
                        }
                        trace.instructions[i] = instruction;
                        // don't count cycles in trace for RVV instructions. They
                        // will be counted in slow path.
                        trace.cycles += self
                            .machine
                            .instruction_cycle_func()
                            .as_ref()
                            .map(|f| f(instruction, 0, 0, true))
                            .unwrap_or(0);
                        let opcode = extract_opcode(instruction);
                        // Here we are calculating the absolute address used in direct threading
                        // from label offsets.
                        trace.thread[i] = unsafe {
                            u64::from(
                                *(ckb_vm_asm_labels as *const u32).offset(opcode as u8 as isize),
                            ) + (ckb_vm_asm_labels as *const u32 as u64)
                        };
                        i += 1;
                        if end_instruction {
                            break;
                        }
                    }
                    trace.instructions[i] = blank_instruction(OP_CUSTOM_TRACE_END);
                    trace.thread[i] = unsafe {
                        u64::from(
                            *(ckb_vm_asm_labels as *const u32).offset(OP_CUSTOM_TRACE_END as isize),
                        ) + (ckb_vm_asm_labels as *const u32 as u64)
                    };
                    trace.address = pc;
                    trace.length = (current_pc - pc) as u8;

                    if trace.slowpath != 0 && self.v_traces.get(&pc).is_none() {
                        if let Some(v_trace) =
                            Self::try_build_v_trace(&trace, &handle_function_list)
                        {
                            self.v_traces.put(pc, v_trace);
                        }
                    }
                    self.machine.inner_mut().inner.traces[slot] = trace;
                }
                RET_ECALL => self.machine.ecall()?,
                RET_EBREAK => self.machine.ebreak()?,
                RET_DYNAMIC_JUMP => (),
                RET_MAX_CYCLES_EXCEEDED => return Err(Error::CyclesExceeded),
                RET_CYCLES_OVERFLOW => return Err(Error::CyclesOverflow),
                RET_OUT_OF_BOUND => return Err(Error::MemOutOfBound),
                RET_INVALID_PERMISSION => return Err(Error::MemWriteOnExecutablePage),
                RET_SLOWPATH_TRACE => loop {
                    let pc = *self.machine.pc();
                    let slot = calculate_slot(pc);
                    let slowpath = self.machine.inner.inner.traces[slot].slowpath;

                    if slowpath == 0 {
                        break;
                    }
                    let cycles = self.machine.inner.inner.traces[slot].cycles;
                    self.machine.add_cycles(cycles)?;

                    if let Some(v_trace) = self.v_traces.get(&pc) {
                        // Optimized VTrace
                        setup_pc_for_trace(
                            &mut self.machine,
                            v_trace.code_length,
                            v_trace.last_inst_length,
                        );
                        for action in &v_trace.actions {
                            action(&mut self.machine)?;
                        }
                        self.machine.commit_pc();
                    } else {
                        // VTrace is not avaiable, fallback to plain executing mode
                        let code_length = self.machine.inner.inner.traces[slot].length;
                        let last_inst_length =
                            self.machine.inner.inner.traces[slot].last_inst_length;

                        setup_pc_for_trace(&mut self.machine, code_length, last_inst_length);
                        for instruction in self.machine.inner.inner.traces[slot].instructions {
                            if instruction == blank_instruction(OP_CUSTOM_TRACE_END) {
                                break;
                            }
                            execute_instruction(
                                &mut self.machine,
                                &handle_function_list_2,
                                instruction,
                            )?;
                        }
                        self.machine.commit_pc();
                    }
                },
                _ => return Err(Error::Asm(result)),
            }
        }
        println!("Total v traces: {}", self.v_traces.len());
        Ok(self.machine.exit_code())
    }

    pub fn try_build_v_trace(
        trace: &Trace,
        handle_function_list: &[Option<HandleFunction<VInferMachine>>],
    ) -> Option<VTrace<'a>> {
        let mut v_trace = VTrace {
            address: trace.address,
            code_length: trace.length,
            ..Default::default()
        };

        let mut i = 0;
        let mut first_v_processed = false;
        let mut infer_machine = VInferMachine::default();
        while extract_opcode(trace.instructions[i]) != OP_CUSTOM_TRACE_END {
            let inst = trace.instructions[i];
            i += 1;
            v_trace.last_inst_length = instruction_length(inst);

            let opcode = extract_opcode(inst);
            if !is_slowpath_instruction(inst) {
                match opcode {
                    insts::OP_UNLOADED => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_unloaded(m, inst)
                    })),
                    insts::OP_ADD => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_add(m, inst)
                    })),
                    insts::OP_ADDI => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_addi(m, inst)
                    })),
                    insts::OP_ADDIW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_addiw(m, inst)
                    })),
                    insts::OP_ADDW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_addw(m, inst)
                    })),
                    insts::OP_AND => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_and(m, inst)
                    })),
                    insts::OP_ANDI => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_andi(m, inst)
                    })),
                    insts::OP_DIV => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_div(m, inst)
                    })),
                    insts::OP_DIVU => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_divu(m, inst)
                    })),
                    insts::OP_DIVUW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_divuw(m, inst)
                    })),
                    insts::OP_DIVW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_divw(m, inst)
                    })),
                    insts::OP_FENCE => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_fence(m, inst)
                    })),
                    insts::OP_FENCEI => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_fencei(m, inst)
                    })),
                    insts::OP_LB => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_lb(m, inst)
                    })),
                    insts::OP_LBU => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_lbu(m, inst)
                    })),
                    insts::OP_LD => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_ld(m, inst)
                    })),
                    insts::OP_LH => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_lh(m, inst)
                    })),
                    insts::OP_LHU => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_lhu(m, inst)
                    })),
                    insts::OP_LUI => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_lui(m, inst)
                    })),
                    insts::OP_LW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_lw(m, inst)
                    })),
                    insts::OP_LWU => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_lwu(m, inst)
                    })),
                    insts::OP_MUL => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_mul(m, inst)
                    })),
                    insts::OP_MULH => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_mulh(m, inst)
                    })),
                    insts::OP_MULHSU => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_mulhsu(m, inst)
                    })),
                    insts::OP_MULHU => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_mulhu(m, inst)
                    })),
                    insts::OP_MULW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_mulw(m, inst)
                    })),
                    insts::OP_OR => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_or(m, inst)
                    })),
                    insts::OP_ORI => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_ori(m, inst)
                    })),
                    insts::OP_REM => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_rem(m, inst)
                    })),
                    insts::OP_REMU => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_remu(m, inst)
                    })),
                    insts::OP_REMUW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_remuw(m, inst)
                    })),
                    insts::OP_REMW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_remw(m, inst)
                    })),
                    insts::OP_SB => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sb(m, inst)
                    })),
                    insts::OP_SD => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sd(m, inst)
                    })),
                    insts::OP_SH => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sh(m, inst)
                    })),
                    insts::OP_SLL => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sll(m, inst)
                    })),
                    insts::OP_SLLI => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_slli(m, inst)
                    })),
                    insts::OP_SLLIW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_slliw(m, inst)
                    })),
                    insts::OP_SLLW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sllw(m, inst)
                    })),
                    insts::OP_SLT => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_slt(m, inst)
                    })),
                    insts::OP_SLTI => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_slti(m, inst)
                    })),
                    insts::OP_SLTIU => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sltiu(m, inst)
                    })),
                    insts::OP_SLTU => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sltu(m, inst)
                    })),
                    insts::OP_SRA => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sra(m, inst)
                    })),
                    insts::OP_SRAI => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_srai(m, inst)
                    })),
                    insts::OP_SRAIW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sraiw(m, inst)
                    })),
                    insts::OP_SRAW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sraw(m, inst)
                    })),
                    insts::OP_SRL => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_srl(m, inst)
                    })),
                    insts::OP_SRLI => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_srli(m, inst)
                    })),
                    insts::OP_SRLIW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_srliw(m, inst)
                    })),
                    insts::OP_SRLW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_srlw(m, inst)
                    })),
                    insts::OP_SUB => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sub(m, inst)
                    })),
                    insts::OP_SUBW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_subw(m, inst)
                    })),
                    insts::OP_SW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sw(m, inst)
                    })),
                    insts::OP_XOR => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_xor(m, inst)
                    })),
                    insts::OP_XORI => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_xori(m, inst)
                    })),
                    insts::OP_ADDUW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_adduw(m, inst)
                    })),
                    insts::OP_ANDN => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_andn(m, inst)
                    })),
                    insts::OP_BCLR => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_bclr(m, inst)
                    })),
                    insts::OP_BCLRI => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_bclri(m, inst)
                    })),
                    insts::OP_BEXT => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_bext(m, inst)
                    })),
                    insts::OP_BEXTI => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_bexti(m, inst)
                    })),
                    insts::OP_BINV => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_binv(m, inst)
                    })),
                    insts::OP_BINVI => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_binvi(m, inst)
                    })),
                    insts::OP_BSET => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_bset(m, inst)
                    })),
                    insts::OP_BSETI => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_bseti(m, inst)
                    })),
                    insts::OP_CLMUL => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_clmul(m, inst)
                    })),
                    insts::OP_CLMULH => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_clmulh(m, inst)
                    })),
                    insts::OP_CLMULR => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_clmulr(m, inst)
                    })),
                    insts::OP_CLZ => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_clz(m, inst)
                    })),
                    insts::OP_CLZW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_clzw(m, inst)
                    })),
                    insts::OP_CPOP => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_cpop(m, inst)
                    })),
                    insts::OP_CPOPW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_cpopw(m, inst)
                    })),
                    insts::OP_CTZ => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_ctz(m, inst)
                    })),
                    insts::OP_CTZW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_ctzw(m, inst)
                    })),
                    insts::OP_MAX => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_max(m, inst)
                    })),
                    insts::OP_MAXU => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_maxu(m, inst)
                    })),
                    insts::OP_MIN => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_min(m, inst)
                    })),
                    insts::OP_MINU => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_minu(m, inst)
                    })),
                    insts::OP_ORCB => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_orcb(m, inst)
                    })),
                    insts::OP_ORN => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_orn(m, inst)
                    })),
                    insts::OP_REV8 => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_rev8(m, inst)
                    })),
                    insts::OP_ROL => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_rol(m, inst)
                    })),
                    insts::OP_ROLW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_rolw(m, inst)
                    })),
                    insts::OP_ROR => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_ror(m, inst)
                    })),
                    insts::OP_RORI => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_rori(m, inst)
                    })),
                    insts::OP_RORIW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_roriw(m, inst)
                    })),
                    insts::OP_RORW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_rorw(m, inst)
                    })),
                    insts::OP_SEXTB => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sextb(m, inst)
                    })),
                    insts::OP_SEXTH => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sexth(m, inst)
                    })),
                    insts::OP_SH1ADD => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sh1add(m, inst)
                    })),
                    insts::OP_SH1ADDUW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sh1adduw(m, inst)
                    })),
                    insts::OP_SH2ADD => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sh2add(m, inst)
                    })),
                    insts::OP_SH2ADDUW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sh2adduw(m, inst)
                    })),
                    insts::OP_SH3ADD => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sh3add(m, inst)
                    })),
                    insts::OP_SH3ADDUW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sh3adduw(m, inst)
                    })),
                    insts::OP_SLLIUW => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_slliuw(m, inst)
                    })),
                    insts::OP_XNOR => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_xnor(m, inst)
                    })),
                    insts::OP_ZEXTH => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_zexth(m, inst)
                    })),
                    insts::OP_WIDE_MUL => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_wide_mul(m, inst)
                    })),
                    insts::OP_WIDE_MULU => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_wide_mulu(m, inst)
                    })),
                    insts::OP_WIDE_MULSU => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_wide_mulsu(m, inst)
                    })),
                    insts::OP_WIDE_DIV => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_wide_div(m, inst)
                    })),
                    insts::OP_WIDE_DIVU => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_wide_divu(m, inst)
                    })),
                    insts::OP_LD_SIGN_EXTENDED_32_CONSTANT => {
                        v_trace.actions.push(Box::new(move |m| {
                            crate::instructions::execute::handle_ld_sign_extended_32_constant(
                                m, inst,
                            )
                        }))
                    }
                    insts::OP_ADC => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_adc(m, inst)
                    })),
                    insts::OP_SBB => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_sbb(m, inst)
                    })),
                    insts::OP_CUSTOM_LOAD_IMM => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_custom_load_imm(m, inst)
                    })),
                    insts::OP_AUIPC => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_auipc(m, inst)
                    })),
                    insts::OP_BEQ => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_beq(m, inst)
                    })),
                    insts::OP_BGE => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_bge(m, inst)
                    })),
                    insts::OP_BGEU => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_bgeu(m, inst)
                    })),
                    insts::OP_BLT => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_blt(m, inst)
                    })),
                    insts::OP_BLTU => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_bltu(m, inst)
                    })),
                    insts::OP_BNE => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_bne(m, inst)
                    })),
                    insts::OP_EBREAK => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_ebreak(m, inst)
                    })),
                    insts::OP_ECALL => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_ecall(m, inst)
                    })),
                    insts::OP_JAL => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_jal(m, inst)
                    })),
                    insts::OP_JALR => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_jalr(m, inst)
                    })),
                    insts::OP_FAR_JUMP_REL => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_far_jump_rel(m, inst)
                    })),
                    insts::OP_FAR_JUMP_ABS => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_far_jump_abs(m, inst)
                    })),
                    insts::OP_CUSTOM_TRACE_END => v_trace.actions.push(Box::new(move |m| {
                        crate::instructions::execute::handle_custom_trace_end(m, inst)
                    })),
                    _ => panic!(
                        "Unexpected IMC instruction: {}",
                        insts::instruction_opcode_name(opcode)
                    ),
                    // _ => return None,
                }
                continue;
            }
            if !first_v_processed {
                // The first V instruction must be vsetvli
                // so as to guard against vl/vtype values.
                if ![insts::OP_VSETVLI, insts::OP_VSETIVLI].contains(&opcode) {
                    return None;
                }
                first_v_processed = true;
            }
            match opcode {
                insts::OP_VSETIVLI => {
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    v_trace
                        .actions
                        .push(Box::new(move |m: &mut CM<'a>| handle_vsetivli(m, inst)));
                }
                insts::OP_VSETVLI => {
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    v_trace
                        .actions
                        .push(Box::new(move |m: &mut CM<'a>| handle_vsetvli(m, inst)));
                }
                insts::OP_VLSE256_V => {
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    v_trace
                        .actions
                        .push(Box::new(move |m: &mut CM<'a>| handle_vlse256(m, inst)));
                }
                insts::OP_VLE8_V => {
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    v_trace
                        .actions
                        .push(Box::new(move |m: &mut CM<'a>| handle_vle8(m, inst)));
                }
                insts::OP_VLE256_V => {
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    v_trace
                        .actions
                        .push(Box::new(move |m: &mut CM<'a>| handle_vle256(m, inst)));
                }
                insts::OP_VLE512_V => {
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    v_trace.actions.push(Box::new(move |m: &mut CM<'a>| {
                        crate::instructions::execute::handle_vle512_v(m, inst)
                    }));
                }
                insts::OP_VSE256_V => {
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    v_trace
                        .actions
                        .push(Box::new(move |m: &mut CM<'a>| handle_vse256(m, inst)));
                }
                insts::OP_VSE512_V => {
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    v_trace.actions.push(Box::new(move |m: &mut CM<'a>| {
                        crate::instructions::execute::handle_vse512_v(m, inst)
                    }));
                }
                insts::OP_VLUXEI8_V => {
                    let sew = infer_machine.vsew();
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    match sew {
                        256 => {
                            v_trace
                                .actions
                                .push(Box::new(move |m: &mut CM<'a>| handle_vluxei8_256(m, inst)));
                        }
                        _ => panic!("Unsupported vluxei8.vv with sew: {}", infer_machine.vsew()),
                        // _ => return None,
                    }
                }
                insts::OP_VADD_VV => {
                    let sew = infer_machine.vsew();
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    match sew {
                        256 => {
                            v_trace
                                .actions
                                .push(Box::new(move |m: &mut CM<'a>| handle_vadd_256(m, inst)));
                        }
                        512 => {
                            v_trace
                                .actions
                                .push(Box::new(move |m: &mut CM<'a>| handle_vadd_512(m, inst)));
                        }
                        _ => panic!("Unsupported vwadc.vv with sew: {}", infer_machine.vsew()),
                        // _ => return None,
                    }
                }
                insts::OP_VMADC_VV => {
                    let sew = infer_machine.vsew();
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    match sew {
                        256 => {
                            v_trace
                                .actions
                                .push(Box::new(move |m: &mut CM<'a>| handle_vmadc_256(m, inst)));
                        }
                        512 => {
                            v_trace
                                .actions
                                .push(Box::new(move |m: &mut CM<'a>| handle_vmadc_512(m, inst)));
                        }
                        _ => panic!("Unsupported vmadc.vv with sew: {}", infer_machine.vsew()),
                        // _ => return None,
                    }
                }
                insts::OP_VSUB_VV => {
                    let sew = infer_machine.vsew();
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    match sew {
                        256 => {
                            v_trace
                                .actions
                                .push(Box::new(move |m: &mut CM<'a>| handle_vsub_256(m, inst)));
                        }
                        512 => {
                            v_trace.actions.push(Box::new(move |m: &mut CM<'a>| {
                                crate::instructions::execute::handle_vsub_vv(m, inst)
                            }));
                        }
                        _ => panic!("Unsupported vsub.vv with sew: {}", infer_machine.vsew()),
                        // _ => return None,
                    }
                }
                insts::OP_VMSBC_VV => {
                    let sew = infer_machine.vsew();
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    match sew {
                        256 => {
                            v_trace
                                .actions
                                .push(Box::new(move |m: &mut CM<'a>| handle_vmsbc_256(m, inst)));
                        }
                        _ => panic!("Unsupported vmsbc.vv with sew: {}", infer_machine.vsew()),
                        // _ => return None,
                    }
                }
                insts::OP_VWMULU_VV => {
                    let sew = infer_machine.vsew();
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    match sew {
                        256 => {
                            v_trace
                                .actions
                                .push(Box::new(move |m: &mut CM<'a>| handle_vmmulu_256(m, inst)));
                        }
                        512 => {
                            v_trace.actions.push(Box::new(move |m: &mut CM<'a>| {
                                crate::instructions::execute::handle_vwmulu_vv(m, inst)
                            }));
                        }
                        _ => panic!("Unsupported vwmulu.vv with sew: {}", infer_machine.vsew()),
                        // _ => return None,
                    }
                }
                insts::OP_VMUL_VV => {
                    let sew = infer_machine.vsew();
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    match sew {
                        256 => {
                            v_trace
                                .actions
                                .push(Box::new(move |m: &mut CM<'a>| handle_vmul_256(m, inst)));
                        }
                        _ => panic!("Unsupported vmulu.vv with sew: {}", infer_machine.vsew()),
                        // _ => return None,
                    }
                }
                insts::OP_VXOR_VV => {
                    let sew = infer_machine.vsew();
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    match sew {
                        256 => {
                            v_trace
                                .actions
                                .push(Box::new(move |m: &mut CM<'a>| handle_vxor_256(m, inst)));
                        }
                        _ => panic!("Unsupported vxor.vv with sew: {}", infer_machine.vsew()),
                        // _ => return None,
                    }
                }
                insts::OP_VNSRL_WX => {
                    let sew = infer_machine.vsew();
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    match sew {
                        256 => {
                            v_trace
                                .actions
                                .push(Box::new(move |m: &mut CM<'a>| handle_vnsrl_256(m, inst)));
                        }
                        512 => {
                            v_trace.actions.push(Box::new(move |m: &mut CM<'a>| {
                                crate::instructions::execute::handle_vnsrl_wx(m, inst)
                            }));
                        }
                        _ => panic!("Unsupported vnsrl.wx with sew: {}", infer_machine.vsew()),
                        // _ => return None,
                    }
                }
                insts::OP_VMANDNOT_MM => {
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    v_trace
                        .actions
                        .push(Box::new(move |m: &mut CM<'a>| handle_vmandnot(m, inst)));
                }
                insts::OP_VMXOR_MM => {
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    v_trace
                        .actions
                        .push(Box::new(move |m: &mut CM<'a>| handle_vmxor(m, inst)));
                }
                insts::OP_VMERGE_VVM => {
                    let sew = infer_machine.vsew();
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    match sew {
                        256 => {
                            v_trace
                                .actions
                                .push(Box::new(move |m: &mut CM<'a>| handle_vmerge_256(m, inst)));
                        }
                        _ => panic!("Unsupported vmerge.vvm with sew: {}", infer_machine.vsew()),
                        // _ => return None,
                    }
                }
                insts::OP_VSLL_VX => {
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    v_trace.actions.push(Box::new(move |m: &mut CM<'a>| {
                        crate::instructions::execute::handle_vsll_vx(m, inst)
                    }));
                }
                insts::OP_VSRL_VX => {
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    v_trace.actions.push(Box::new(move |m: &mut CM<'a>| {
                        crate::instructions::execute::handle_vsrl_vx(m, inst)
                    }));
                }
                insts::OP_VWMACCU_VV => {
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    v_trace.actions.push(Box::new(move |m: &mut CM<'a>| {
                        crate::instructions::execute::handle_vwmaccu_vv(m, inst)
                    }));
                }
                insts::OP_VMSLEU_VV => {
                    execute(&mut infer_machine, handle_function_list, inst).ok()?;
                    v_trace.actions.push(Box::new(move |m: &mut CM<'a>| {
                        crate::instructions::execute::handle_vmsleu_vv(m, inst)
                    }));
                }
                _ => panic!(
                    "Unsupported v op: {}, sew now: {}",
                    insts::instruction_opcode_name(opcode),
                    infer_machine.vsew(),
                ),
                // _ => return None,
            };
        }
        println!(
            "Built trace: {:x}, length: {}",
            v_trace.address,
            v_trace.actions.len()
        );
        Some(v_trace)
    }
}

fn handle_vsetvli<'a>(m: &mut CM<'a>, inst: Instruction) -> Result<(), Error> {
    let i = Itype(inst);
    common::set_vl(
        m,
        i.rd(),
        i.rs1(),
        m.registers()[i.rs1()].to_u64(),
        i.immediate_u() as u64,
    )
}

fn handle_vsetivli<'a>(m: &mut CM<'a>, inst: Instruction) -> Result<(), Error> {
    let i = Itype(inst);
    common::set_vl(m, i.rd(), 33, i.rs1() as u64, i.immediate_u() as u64)
}

fn handle_vlse256<'a>(m: &mut CM<'a>, inst: Instruction) -> Result<(), Error> {
    let i = VXtype(inst);
    let addr = m.registers()[i.rs1()].to_u64();
    let stride = m.registers()[i.vs2()].to_u64();

    let vl = m.vl();
    if stride == 0 && vl > 0 && i.vm() != 0 {
        // When stride is zero, we will be repeatedly loading from the same memory
        // address again and again, we could save a few memory checking here.
        // TODO: build a fan out function here, or better, combine with vluxei8
        // to eliminate unnecessary memory checking
        check_memory_inited(&mut m.inner.inner, addr, 32)?;

        for j in 0..vl {
            let i0 = i.vd() * (VLEN >> 3) + 32 * (j as usize);
            let i1 = i0 + 32;

            m.inner.inner.register_file[i0..i1]
                .copy_from_slice(&m.inner.inner.memory[addr as usize..addr as usize + 32]);
        }
    } else {
        for j in 0..vl {
            if i.vm() == 0 && !m.get_bit(0, j as usize) {
                continue;
            }

            m.inner.mem_to_v(
                i.vd(),
                32 << 3,
                j as usize,
                1,
                stride.wrapping_mul(j).wrapping_add(addr),
            )?;
        }
    }
    Ok(())
}

fn handle_vle8<'a>(m: &mut CM<'a>, inst: Instruction) -> Result<(), Error> {
    let i = VXtype(inst);
    let addr = m.registers()[i.rs1()].to_u64();
    let stride = 1u64;

    if i.vm() != 0 {
        m.inner.mem_to_v(i.vd(), 1 << 3, 0, m.vl() as usize, addr)?;
    } else {
        for j in 0..m.vl() {
            if !m.get_bit(0, j as usize) {
                continue;
            }
            m.inner.mem_to_v(
                i.vd(),
                1 << 3,
                j as usize,
                1,
                stride.wrapping_mul(j).wrapping_add(addr),
            )?;
        }
    }
    Ok(())
}

fn handle_vle256<'a>(m: &mut CM<'a>, inst: Instruction) -> Result<(), Error> {
    let i = VXtype(inst);
    let addr = m.registers()[i.rs1()].to_u64();
    let stride = 32u64;

    if i.vm() != 0 {
        m.inner
            .mem_to_v(i.vd(), 32 << 3, 0, m.vl() as usize, addr)?;
    } else {
        for j in 0..m.vl() {
            if !m.get_bit(0, j as usize) {
                continue;
            }
            m.inner.mem_to_v(
                i.vd(),
                32 << 3,
                j as usize,
                1,
                stride.wrapping_mul(j).wrapping_add(addr),
            )?;
        }
    }
    Ok(())
}

fn handle_vluxei8_256<'a>(m: &mut CM<'a>, inst: Instruction) -> Result<(), Error> {
    let sew = 256;
    let i = VXtype(inst);
    let addr = m.registers()[i.rs1()].to_u64();
    for j in 0..m.vl() as usize {
        if i.vm() == 0 && !m.get_bit(0, j) {
            continue;
        }
        let offset = E8::get(m.inner.v_ref(i.vs2(), 8, j, 1)).u64();
        m.inner
            .mem_to_v(i.vd(), sew, j, 1, addr.wrapping_add(offset))?;
    }
    Ok(())
}

fn handle_vse256<'a>(m: &mut CM<'a>, inst: Instruction) -> Result<(), Error> {
    let i = VXtype(inst);
    let addr = m.registers()[i.rs1()].to_u64();
    let stride = 32u64;

    if i.vm() != 0 {
        m.inner
            .v_to_mem(i.vd(), 32 << 3, 0, m.vl() as usize, addr)?;
    } else {
        for j in 0..m.vl() {
            if !m.get_bit(0, j as usize) {
                continue;
            }
            m.inner.v_to_mem(
                i.vd(),
                32 << 3,
                j as usize,
                1,
                stride.wrapping_mul(j).wrapping_add(addr),
            )?;
        }
    }
    Ok(())
}

fn handle_vadd_256<'a>(m: &mut CM<'a>, inst: Instruction) -> Result<(), Error> {
    let i = VVtype(inst);
    if i.vm() != 0 {
        let vl = m.vl() as usize;
        utils::wrapping_add_256(
            m.inner.v_ref(i.vs1(), 256, 0, vl).as_ptr(),
            m.inner.v_ref(i.vs2(), 256, 0, vl).as_ptr(),
            m.inner.v_mut(i.vd(), 256, 0, vl).as_mut_ptr(),
            vl,
        );
    } else {
        for j in 0..m.vl() as usize {
            if !m.get_bit(0, j) {
                continue;
            }

            utils::wrapping_add_256(
                m.inner.v_ref(i.vs1(), 256, j, 1).as_ptr(),
                m.inner.v_ref(i.vs2(), 256, j, 1).as_ptr(),
                m.inner.v_mut(i.vd(), 256, j, 1).as_mut_ptr(),
                1,
            );
        }
    }
    Ok(())
}

fn handle_vadd_512<'a>(m: &mut CM<'a>, inst: Instruction) -> Result<(), Error> {
    let i = VVtype(inst);
    if i.vm() != 0 {
        let vl = m.vl() as usize;
        utils::wrapping_add_512(
            m.inner.v_ref(i.vs1(), 512, 0, vl).as_ptr(),
            m.inner.v_ref(i.vs2(), 512, 0, vl).as_ptr(),
            m.inner.v_mut(i.vd(), 512, 0, vl).as_mut_ptr(),
            vl,
        );
    } else {
        for j in 0..m.vl() as usize {
            if !m.get_bit(0, j) {
                continue;
            }

            utils::wrapping_add_512(
                m.inner.v_ref(i.vs1(), 512, j, 1).as_ptr(),
                m.inner.v_ref(i.vs2(), 512, j, 1).as_ptr(),
                m.inner.v_mut(i.vd(), 512, j, 1).as_mut_ptr(),
                1,
            );
        }
    }
    Ok(())
}

fn handle_vmadc_256<'a>(m: &mut CM<'a>, inst: Instruction) -> Result<(), Error> {
    let sew = 256;
    let i = VVtype(inst);
    for j in 0..m.vl() as usize {
        if i.vm() == 0 && !m.get_bit(0, j) {
            continue;
        }
        let c = utils::madc_256(
            m.inner.v_ref(i.vs2(), sew, j, 1).as_ptr(),
            m.inner.v_ref(i.vs1(), sew, j, 1).as_ptr(),
        );
        if c {
            m.set_bit(i.vd(), j);
        } else {
            m.clr_bit(i.vd(), j);
        };
    }
    Ok(())
}

fn handle_vmadc_512<'a>(m: &mut CM<'a>, inst: Instruction) -> Result<(), Error> {
    let sew = 512;
    let i = VVtype(inst);
    for j in 0..m.vl() as usize {
        if i.vm() == 0 && !m.get_bit(0, j) {
            continue;
        }
        let c = utils::madc_512(
            m.inner.v_ref(i.vs2(), sew, j, 1).as_ptr(),
            m.inner.v_ref(i.vs1(), sew, j, 1).as_ptr(),
        );
        if c {
            m.set_bit(i.vd(), j);
        } else {
            m.clr_bit(i.vd(), j);
        };
    }
    Ok(())
}

fn handle_vsub_256<'a>(m: &mut CM<'a>, inst: Instruction) -> Result<(), Error> {
    let sew = 256;
    let i = VVtype(inst);
    if i.vm() != 0 {
        let vl = m.vl() as usize;
        utils::wrapping_sub_256(
            m.inner.v_ref(i.vs2(), sew, 0, vl).as_ptr(),
            m.inner.v_ref(i.vs1(), sew, 0, vl).as_ptr(),
            m.inner.v_mut(i.vd(), sew, 0, vl).as_mut_ptr(),
            vl,
        );
    } else {
        for j in 0..m.vl() as usize {
            if !m.get_bit(0, j) {
                continue;
            }

            utils::wrapping_sub_256(
                m.inner.v_ref(i.vs2(), sew, j, 1).as_ptr(),
                m.inner.v_ref(i.vs1(), sew, j, 1).as_ptr(),
                m.inner.v_mut(i.vd(), sew, j, 1).as_mut_ptr(),
                1,
            );
        }
    }
    Ok(())
}

fn handle_vmsbc_256<'a>(m: &mut CM<'a>, inst: Instruction) -> Result<(), Error> {
    let sew = 256;
    let i = VVtype(inst);
    for j in 0..m.vl() as usize {
        if i.vm() == 0 && !m.get_bit(0, j) {
            continue;
        }
        let c = utils::msbc_256(
            m.inner.v_ref(i.vs2(), sew, j, 1).as_ptr(),
            m.inner.v_ref(i.vs1(), sew, j, 1).as_ptr(),
        );
        if c {
            m.set_bit(i.vd(), j);
        } else {
            m.clr_bit(i.vd(), j);
        };
    }
    Ok(())
}

fn handle_vmmulu_256<'a>(m: &mut CM<'a>, inst: Instruction) -> Result<(), Error> {
    let sew = 256;
    let i = VVtype(inst);
    if i.vm() != 0 && i.vs1() != i.vs2() && i.vs1() != i.vd() && i.vs2() != i.vd() {
        let vl = m.vl() as usize;
        utils::widening_mul_256_non_overlapping(
            m.inner.v_ref(i.vs1(), sew, 0, vl).as_ptr(),
            m.inner.v_ref(i.vs2(), sew, 0, vl).as_ptr(),
            m.inner.v_mut(i.vd(), sew * 2, 0, vl).as_mut_ptr(),
            vl,
        );
    } else {
        for j in 0..m.vl() as usize {
            if i.vm() == 0 && !m.get_bit(0, j as usize) {
                continue;
            }

            let mut c: [u8; 64] = unsafe { MaybeUninit::uninit().assume_init() };

            utils::widening_mul_256_non_overlapping(
                m.inner.v_ref(i.vs1(), sew, j, 1).as_ptr(),
                m.inner.v_ref(i.vs2(), sew, j, 1).as_ptr(),
                c.as_mut_ptr(),
                1,
            );

            m.inner.v_mut(i.vd(), sew * 2, j, 1).copy_from_slice(&c);
        }
    }
    Ok(())
}

fn handle_vmul_256<'a>(m: &mut CM<'a>, inst: Instruction) -> Result<(), Error> {
    let sew = 256;
    let i = VVtype(inst);
    if i.vm() != 0 {
        let vl = m.vl() as usize;
        utils::wrapping_mul_256(
            m.inner.v_ref(i.vs2(), sew, 0, vl).as_ptr(),
            m.inner.v_ref(i.vs1(), sew, 0, vl).as_ptr(),
            m.inner.v_mut(i.vd(), sew, 0, vl).as_mut_ptr(),
            vl,
        )
    } else {
        for j in 0..m.vl() as usize {
            if !m.get_bit(0, j as usize) {
                continue;
            }
            utils::wrapping_mul_256(
                m.inner.v_ref(i.vs2(), sew, j, 1).as_ptr(),
                m.inner.v_ref(i.vs1(), sew, j, 1).as_ptr(),
                m.inner.v_mut(i.vd(), sew, j, 1).as_mut_ptr(),
                1,
            );
        }
    }
    Ok(())
}

fn handle_vxor_256<'a>(m: &mut CM<'a>, inst: Instruction) -> Result<(), Error> {
    let sew = 256;
    let i = VVtype(inst);
    for j in 0..m.vl() as usize {
        if i.vm() == 0 && !m.get_bit(0, j as usize) {
            continue;
        }
        let b = E256::get(m.inner.v_ref(i.vs2(), sew, j, 1));
        let a = E256::get(m.inner.v_ref(i.vs1(), sew, j, 1));
        let r = alu::xor(b, a);
        r.put(m.inner.v_mut(i.vd(), sew, j, 1));
    }
    Ok(())
}

fn handle_vnsrl_256<'a>(m: &mut CM<'a>, inst: Instruction) -> Result<(), Error> {
    let sew = 256;
    let i = VXtype(inst);
    if i.vm() != 0 {
        let vl = m.vl() as usize;
        utils::narrowing_right_shift_512(
            m.inner.v_ref(i.vs2(), sew * 2, 0, vl).as_ptr(),
            m.inner.v_mut(i.vd(), sew, 0, vl).as_mut_ptr(),
            m.registers()[i.rs1()].to_u32(),
            vl,
        );
    } else {
        for j in 0..m.vl() as usize {
            if !m.get_bit(0, j) {
                continue;
            }
            utils::narrowing_right_shift_512(
                m.inner.v_ref(i.vs2(), sew * 2, j, 1).as_ptr(),
                m.inner.v_mut(i.vd(), sew, j, 1).as_mut_ptr(),
                m.registers()[i.rs1()].to_u32(),
                1,
            );
        }
    }
    Ok(())
}

fn handle_vmandnot<'a>(m: &mut CM<'a>, inst: Instruction) -> Result<(), Error> {
    let i = VVtype(inst);
    for j in 0..m.vl() as usize {
        let b = m.get_bit(i.vs2(), j);
        let a = m.get_bit(i.vs1(), j);
        if b & !a {
            m.set_bit(i.vd(), j);
        } else {
            m.clr_bit(i.vd(), j);
        }
    }
    Ok(())
}

fn handle_vmxor<'a>(m: &mut CM<'a>, inst: Instruction) -> Result<(), Error> {
    let i = VVtype(inst);
    for j in 0..m.vl() as usize {
        let b = m.get_bit(i.vs2(), j);
        let a = m.get_bit(i.vs1(), j);
        if b ^ a {
            m.set_bit(i.vd(), j);
        } else {
            m.clr_bit(i.vd(), j);
        }
    }
    Ok(())
}

fn handle_vmerge_256<'a>(m: &mut CM<'a>, inst: Instruction) -> Result<(), Error> {
    let sew = 256;
    let i = VVtype(inst);
    for j in 0..m.vl() as usize {
        let mbit = m.get_bit(0, j);
        let src = if mbit { i.vs1() } else { i.vs2() };

        m.inner.v_to_v(src, sew, j, 1, i.vd())?;
    }
    Ok(())
}

// One observation from ckb-vm, is that only the last instruction of each trace
// could read/write pc value. Using this assumption, there is no need to invoke
// update_pc/commit_pc for each individual instruction in a trace. The following
// actions should be enough before running the trace:
// * commit pc to the address before the last instruction in a trace
// * update pc to the address after the whole trace
// This way a huge number of update_pc/commit_pc calls can be saved. Notice this
// very technique applies to both IMC traces, and V traces.
fn setup_pc_for_trace<'a>(m: &mut CM<'a>, code_length: u8, last_inst_length: u8) {
    let pc = *m.pc();
    let trace_end_pc = pc.wrapping_add(code_length as u64);
    let before_last_inst_pc = trace_end_pc.wrapping_sub(last_inst_length as u64);
    m.update_pc(before_last_inst_pc);
    m.commit_pc();
    m.update_pc(trace_end_pc);
}
