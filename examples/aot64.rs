use ckb_vm::instructions::cost_model::instruction_cycles;
use ckb_vm::registers::{A0, A7};
use ckb_vm::{CoreMachine, Memory, Register, SupportMachine, Syscalls};

pub struct CustomSyscall {}

impl<Mac: SupportMachine> Syscalls<Mac> for CustomSyscall {
    fn initialize(&mut self, _machine: &mut Mac) -> Result<(), ckb_vm::error::Error> {
        Ok(())
    }

    fn ecall(&mut self, machine: &mut Mac) -> Result<bool, ckb_vm::error::Error> {
        let code = &machine.registers()[A7];
        if code.to_i32() != 2177 {
            return Ok(false);
        }

        let mut addr = machine.registers()[A0].to_u64();
        let mut buffer = Vec::new();

        loop {
            let byte = machine
                .memory_mut()
                .load8(&Mac::REG::from_u64(addr))?
                .to_u8();
            if byte == 0 {
                break;
            }
            buffer.push(byte);
            addr += 1;
        }

        let s = String::from_utf8(buffer).unwrap();
        println!("{:?}", s);

        Ok(true)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let code = std::fs::read(&args[1])?.into();

    let mut aot_machine = ckb_vm::machine::aot::AotCompilingMachine::load(
        &code,
        Some(Box::new(instruction_cycles)),
        ckb_vm::ISA_IMC | ckb_vm::ISA_V,
        ckb_vm::machine::VERSION1,
    )
    .unwrap();
    let aot_code = aot_machine.compile().unwrap();

    let asm_core = ckb_vm::machine::asm::AsmCoreMachine::new(
        ckb_vm::ISA_IMC | ckb_vm::ISA_B | ckb_vm::ISA_V,
        ckb_vm::machine::VERSION1,
        u64::MAX,
    );
    let core =
        ckb_vm::DefaultMachineBuilder::<Box<ckb_vm::machine::asm::AsmCoreMachine>>::new(asm_core)
            .instruction_cycle_func(Box::new(instruction_cycles))
            .syscall(Box::new(CustomSyscall {}))
            .build();
    let mut machine = ckb_vm::machine::asm::AsmMachine::new(core, Some(&aot_code));

    machine.load_program(&code, &vec!["main".into()]).unwrap();

    let exit = machine.run();
    let cycles = machine.machine.cycles();
    println!(
        "aot exit={:?} cycles={:?} r[a1]={:?}",
        exit,
        cycles,
        machine.machine.registers()[ckb_vm::registers::A1]
    );

    std::process::exit(exit.unwrap() as i32);
}