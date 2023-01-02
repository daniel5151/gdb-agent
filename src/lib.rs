#![allow(clippy::bool_to_int_with_if)]
#![deny(missing_debug_implementations)]
#![warn(missing_docs)]
#![cfg_attr(not(feature = "alloc"), no_std)]

//! TODO: crate docs

use core::convert::TryFrom;
use core::convert::TryInto;
use managed::ManagedSlice;
use num_enum::TryFromPrimitive;

type Result<T> = core::result::Result<T, AgentError>;

/// Errors which may occur during gdb agent bytecode evaluation
#[derive(Debug)]
pub enum AgentError {
    /// Evaluating opcode at `pos` resulted in a division by zero
    DivideByZero(Opcode, u64),
    /// No value on the evaluation stack when evaluating opcode at `pos`
    NoValueOnStack(Opcode, u64),
    /// The index is not a valid index for the evaluation stack when evaluating opcode at `pos`
    PickIndexOutOfRange(Opcode, u64, usize),
    /// Unrecognized opcode value at `pos`
    UnrecognizedOpcode(u8, u64),
    /// Unexpected end of bytecode stream
    UnexpectedEof,
    /// Bytecode stack overflow
    StackOverflow,
}

impl core::fmt::Display for AgentError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AgentError::DivideByZero(o, pos) => write!(f, "Evaluating the opcode {:?} at {} resulted in a division by zero", o, pos),
            AgentError::NoValueOnStack(o, pos) => write!(f, "No value on the evaluation stack when evaluating opcode {:?} at {}", o, pos),
            AgentError::PickIndexOutOfRange(o, pos, index) => write!(f, "The index {} is not a valid index for the evaluation stack when evaluating opcode {:?} at {}", index, o, pos),
            AgentError::UnrecognizedOpcode(o, pos) => write!(f, "Opcode value '{}' is unknown at {}", o, pos),
            AgentError::UnexpectedEof => write!(f, "Unexpected end of bytecode"),
            AgentError::StackOverflow => write!(f, "Bytecode stack overflow"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AgentError {}

trait ReadBigEndian: Sized {
    fn read_big_endian(b: &[u8], pos: &mut u64) -> Option<Self>;
}

macro_rules! impl_read_big_endian {
    ($n:ty) => {
        impl ReadBigEndian for $n {
            fn read_big_endian(b: &[u8], pos: &mut u64) -> Option<Self> {
                const NUM_SIZE: usize = core::mem::size_of::<$n>();
                let n = b.get((*pos as usize)..((*pos as usize) + NUM_SIZE))?;
                *pos += NUM_SIZE as u64;
                Some(<$n>::from_be_bytes(n.try_into().unwrap()))
            }
        }
    };
}

impl_read_big_endian!(u8);
impl_read_big_endian!(u16);
impl_read_big_endian!(u32);
impl_read_big_endian!(u64);

/// A value produced by a gdb agent expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Value(pub i64);

/// A gdb agent expression opcode.
#[allow(missing_docs)] // self explanatory variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive)]
#[repr(u8)]
pub enum Opcode {
    OpFloat = 0x01,
    OpAdd = 0x02,
    OpSub = 0x03,
    OpMul = 0x04,
    OpDivSigned = 0x05,
    OpDivUnsigned = 0x06,
    OpRemSigned = 0x07,
    OpRemUnsigned = 0x08,
    OpLsh = 0x09,
    OpRshSigned = 0x0a,
    OpRshUnsigned = 0x0b,
    OpTrace = 0x0c,
    OpTraceQuick = 0x0d,
    OpLogNot = 0x0e,
    OpBitAnd = 0x0f,
    OpBitOr = 0x10,
    OpBitXor = 0x11,
    OpBitNot = 0x12,
    OpEqual = 0x13,
    OpLessSigned = 0x14,
    OpLessUnsigned = 0x15,
    OpExt = 0x16,
    OpRef8 = 0x17,
    OpRef16 = 0x18,
    OpRef32 = 0x19,
    OpRef64 = 0x1a,
    OpRefFloat = 0x1b,
    OpRefDouble = 0x1c,
    OpRefLongDouble = 0x1d,
    OpLToD = 0x1e,
    OpDToL = 0x1f,
    OpIfGoto = 0x20,
    OpGoto = 0x21,
    OpConst8 = 0x22,
    OpConst16 = 0x23,
    OpConst32 = 0x24,
    OpConst64 = 0x25,
    OpReg = 0x26,
    OpEnd = 0x27,
    OpDup = 0x28,
    OpPop = 0x29,
    OpZeroExt = 0x2a,
    OpSwap = 0x2b,
    OpGetv = 0x2c,
    OpSetv = 0x2d,
    OpTracev = 0x2e,
    OpTracenz = 0x2f,
    OpTrace16 = 0x30,
    OpPick = 0x32,
    OpRot = 0x33,
    OpPrintf = 0x34,
}

enum State {
    Complete(Value),
    NeedsRegister(u16),
    NeedsMemory { address: Value, size: u8 },
    Continue,
}

#[derive(Debug)]
struct StateMachine<'bc, 's> {
    bytecode: &'bc [u8],
    pos: u64,
    stack: ManagedVec<'s, Value>,
    error: bool,
}

impl<'bc, 's> StateMachine<'bc, 's> {
    #[cfg(feature = "alloc")]
    fn new(bytecode: &'bc [u8]) -> StateMachine<'bc, 'static> {
        StateMachine {
            bytecode,
            pos: 0,
            stack: ManagedVec::new(ManagedSlice::Owned(Vec::new())),
            error: false,
        }
    }

    fn new_fixed(bytecode: &'bc [u8], stack: &'s mut [Value]) -> StateMachine<'bc, 's> {
        StateMachine {
            bytecode,
            pos: 0,
            stack: ManagedVec::new(ManagedSlice::Borrowed(stack)),
            error: false,
        }
    }

    /// Pop a value off the stack, setting the error flag if there is no value
    /// to pop.
    fn pop(&mut self, o: Opcode, pos: u64) -> Result<Value> {
        let ret = self.stack.pop();
        ret.ok_or_else(|| {
            self.error = true;
            AgentError::NoValueOnStack(o, pos)
        })
    }

    /// Pop a pair of operands.
    fn pop_binary_operands(&mut self, o: Opcode, pos: u64) -> Result<(Value, Value)> {
        let b = self.pop(o, pos);
        b.and_then(|b| {
            let a = self.pop(o, pos);
            a.map(|a| (a, b))
        })
    }

    /// Fetch a value of the specified size from the bytecode stream.
    fn fetch<T>(&mut self) -> Result<T>
    where
        T: ReadBigEndian,
    {
        match T::read_big_endian(self.bytecode, &mut self.pos) {
            Some(t) => Ok(t),
            None => {
                self.error = true;
                Err(AgentError::UnexpectedEof)
            }
        }
    }

    /// Add a value to the stack.
    fn push(&mut self, v: Value) -> Result<()> {
        self.stack.push(v).map_err(|_| AgentError::StackOverflow)?;
        Ok(())
    }

    /// Get the `i`th value on the stack, measured from the top, and
    /// push it on the stack.
    fn pick(&mut self, i: usize, o: Opcode, pos: u64) -> Result<()> {
        let stack_length = self.stack.len();
        let index = stack_length
            .checked_sub(1 + i)
            .ok_or(AgentError::PickIndexOutOfRange(o, pos, i))?;
        let v = self.stack.get(index).unwrap();
        self.push(*v)
    }

    /// Ensure the value is nonzero.
    fn nonzero(&mut self, v: Value, o: Opcode, pos: u64) -> Result<Value> {
        if v.0 == 0 {
            return Err(AgentError::DivideByZero(o, pos));
        }

        Ok(v)
    }

    fn step(&mut self) -> Result<State> {
        let pos = self.pos;
        let op = {
            let op_value: u8 = self.fetch()?;
            Opcode::try_from(op_value).map_err(|_| AgentError::UnrecognizedOpcode(op_value, pos))?
        };

        match op {
            Opcode::OpAdd => {
                let (a, b) = self.pop_binary_operands(op, pos)?;
                // Use wrapping functions for C compat.
                self.push(Value(a.0.wrapping_add(b.0)))?
            }
            Opcode::OpSub => {
                let (a, b) = self.pop_binary_operands(op, pos)?;
                // Use wrapping functions for C compat.
                self.push(Value(a.0.wrapping_sub(b.0)))?
            }
            Opcode::OpMul => {
                let (a, b) = self.pop_binary_operands(op, pos)?;
                // Use wrapping functions for C compat.
                self.push(Value(a.0.wrapping_mul(b.0)))?
            }
            Opcode::OpDivSigned => {
                let (a, b) = self.pop_binary_operands(op, pos)?;
                let b = self.nonzero(b, op, pos)?;
                // Use wrapping functions for C compat.
                self.push(Value(a.0.wrapping_div(b.0)))?
            }
            Opcode::OpDivUnsigned => {
                let (a, b) = self.pop_binary_operands(op, pos)?;
                let a = a.0 as u64;
                let b = self.nonzero(b, op, pos)?.0 as u64;
                self.push(Value((a / b) as i64))?
            }
            Opcode::OpRemSigned => {
                let (a, b) = self.pop_binary_operands(op, pos)?;
                let b = self.nonzero(b, op, pos)?;
                // Use wrapping functions for C compat.
                self.push(Value(a.0.wrapping_rem(b.0)))?
            }
            Opcode::OpRemUnsigned => {
                let (a, b) = self.pop_binary_operands(op, pos)?;
                let a = a.0 as u64;
                let b = self.nonzero(b, op, pos)?.0 as u64;
                self.push(Value((a % b) as i64))?
            }
            Opcode::OpLsh => {
                let (a, b) = self.pop_binary_operands(op, pos)?;
                self.push(Value(a.0 << b.0))?
            }
            Opcode::OpRshSigned => {
                let (a, b) = self.pop_binary_operands(op, pos)?;
                self.push(Value(a.0 >> b.0))?
            }
            Opcode::OpRshUnsigned => {
                let (a, b) = self.pop_binary_operands(op, pos)?;
                let a = a.0 as u64;
                let b = b.0 as u64;
                self.push(Value((a >> b) as i64))?
            }
            Opcode::OpLogNot => {
                let a = self.pop(op, pos)?;
                let v = if a.0 == 0 { 1 } else { 0 };
                self.push(Value(v))?
            }
            Opcode::OpBitAnd => {
                let (a, b) = self.pop_binary_operands(op, pos)?;
                self.push(Value(a.0 & b.0))?
            }
            Opcode::OpBitOr => {
                let (a, b) = self.pop_binary_operands(op, pos)?;
                self.push(Value(a.0 | b.0))?
            }
            Opcode::OpBitXor => {
                let (a, b) = self.pop_binary_operands(op, pos)?;
                self.push(Value(a.0 ^ b.0))?
            }
            Opcode::OpBitNot => {
                let a = self.pop(op, pos)?;
                self.push(Value(!a.0))?
            }
            Opcode::OpEqual => {
                let (a, b) = self.pop_binary_operands(op, pos)?;
                let v = if a == b { 1 } else { 0 };
                self.push(Value(v))?
            }
            Opcode::OpLessSigned => {
                let (a, b) = self.pop_binary_operands(op, pos)?;
                let v = if a.0 < b.0 { 1 } else { 0 };
                self.push(Value(v))?
            }
            Opcode::OpLessUnsigned => {
                let (a, b) = self.pop_binary_operands(op, pos)?;
                let a = a.0 as u64;
                let b = b.0 as u64;
                let v = if a < b { 1 } else { 0 };
                self.push(Value(v))?
            }
            Opcode::OpExt => {
                let n = Value(self.fetch::<u8>()? as i64);
                let n = self.nonzero(n, op, pos)?;
                let n = n.0 as u64;
                if n < 64 {
                    let a = self.pop(op, pos)?.0;
                    let n_mask = (1i64 << n) - 1;
                    let sign_bit = (a >> (n - 1)) & 1;
                    self.push(Value((sign_bit * !n_mask) | (a & n_mask)))?
                }
            }
            Opcode::OpZeroExt => {
                let n = self.fetch::<u8>()? as u64;
                if n < 64 {
                    let a = self.pop(op, pos)?.0;
                    let n_mask = (1i64 << n) - 1;
                    self.push(Value(a & n_mask))?
                }
            }
            Opcode::OpDup => self.pick(0, op, pos)?,
            Opcode::OpSwap => {
                let (a, b) = self.pop_binary_operands(op, pos)?;
                self.push(b)?;
                self.push(a)?
            }
            Opcode::OpPop => {
                self.pop(op, pos)?;
            }
            Opcode::OpPick => {
                let i = self.fetch::<u8>()? as usize;
                self.pick(i, op, pos)?
            }
            Opcode::OpRot => {
                let c = self.pop(op, pos)?;
                let b = self.pop(op, pos)?;
                let a = self.pop(op, pos)?;
                self.push(c)?;
                self.push(b)?;
                self.push(a)?
            }
            Opcode::OpIfGoto => {
                let offset = self.fetch::<u16>()? as u64;
                if self.pop(op, pos)?.0 != 0 {
                    self.pos = offset;
                }
            }
            Opcode::OpGoto => {
                let offset = self.fetch::<u16>()? as u64;
                self.pos = offset;
            }
            Opcode::OpConst8 => {
                let value = Value(self.fetch::<u8>()? as i64);
                self.push(value)?
            }
            Opcode::OpConst16 => {
                let value = Value(self.fetch::<u16>()? as i64);
                self.push(value)?
            }
            Opcode::OpConst32 => {
                let value = Value(self.fetch::<u32>()? as i64);
                self.push(value)?
            }
            Opcode::OpConst64 => {
                let value = Value(self.fetch::<u64>()? as i64);
                self.push(value)?
            }
            Opcode::OpReg => {
                let register = self.fetch::<u16>()?;
                return Ok(State::NeedsRegister(register));
            }
            Opcode::OpRef8 => {
                return Ok(State::NeedsMemory {
                    address: self.pop(op, pos)?,
                    size: 1,
                });
            }
            Opcode::OpRef16 => {
                return Ok(State::NeedsMemory {
                    address: self.pop(op, pos)?,
                    size: 2,
                });
            }
            Opcode::OpRef32 => {
                return Ok(State::NeedsMemory {
                    address: self.pop(op, pos)?,
                    size: 4,
                });
            }
            Opcode::OpRef64 => {
                return Ok(State::NeedsMemory {
                    address: self.pop(op, pos)?,
                    size: 8,
                });
            }
            Opcode::OpEnd => {
                return Ok(State::Complete(self.pop(op, pos)?));
            }
            _ => self.error = true,
        }

        Ok(State::Continue)
    }

    fn evaluate(&mut self) -> Result<State> {
        loop {
            match self.step()? {
                State::Continue => continue,
                x => return Ok(x),
            }
        }
    }
}

fn evaluate_internal<'bc, 's>(
    mut state: StateMachine<'bc, 's>,
) -> Result<AgentExpressionResult<'bc, 's>> {
    Ok(match state.evaluate()? {
        State::Complete(v) => AgentExpressionResult::Complete(v),
        State::NeedsRegister(r) => AgentExpressionResult::NeedsRegister {
            register: r,
            expression: AgentExpressionNeedsRegister { state },
        },
        State::NeedsMemory { address, size } => AgentExpressionResult::NeedsMemory {
            address,
            size,
            expression: AgentExpressionNeedsMemory { state },
        },
        State::Continue => unreachable!(),
    })
}

/// Evaluate an agent bytecode expression, using a pre-allocated fixed-size stack.
pub fn evaluate_fixed_stack<'bc, 's>(
    bytecode: &'bc [u8],
    stack: &'s mut [Value],
) -> Result<AgentExpressionResult<'bc, 's>> {
    evaluate_internal(StateMachine::new_fixed(bytecode, stack))
}

/// Evaluate an agent bytecode expression.
#[cfg(feature = "alloc")]
pub fn evaluate<'bc>(bytecode: &'bc [u8]) -> Result<AgentExpressionResult<'bc, 'static>> {
    evaluate_internal(StateMachine::new(bytecode))
}

/// An evaluation which can be resumed once the required register value is provided.
#[derive(Debug)]
pub struct AgentExpressionNeedsRegister<'bc, 's> {
    state: StateMachine<'bc, 's>,
}

impl<'bc, 's> AgentExpressionNeedsRegister<'bc, 's> {
    /// Resume the evaluation with the provided `register_value`.
    pub fn resume_with_register(
        self,
        register_value: Value,
    ) -> Result<AgentExpressionResult<'bc, 's>> {
        let mut state = self.state;
        state.push(register_value)?;
        evaluate_internal(state)
    }
}

/// An evaluation which can be resumed once the required memory value is provided.
#[derive(Debug)]
pub struct AgentExpressionNeedsMemory<'bc, 's> {
    state: StateMachine<'bc, 's>,
}

impl<'bc, 's> AgentExpressionNeedsMemory<'bc, 's> {
    /// Resume the evaluation with the provided `memory_value`.
    pub fn resume_with_memory(self, memory_value: Value) -> Result<AgentExpressionResult<'bc, 's>> {
        let mut state = self.state;
        state.push(memory_value)?;
        evaluate_internal(state)
    }
}

/// The result of evaluating a gdb agent expression.  It may be `Complete`, in
/// which case the value can be retrieved, or it may require additional
/// processing.
#[derive(Debug)]
pub enum AgentExpressionResult<'bc, 's> {
    /// The agent expression needs a register value to continue evaluation.
    NeedsRegister {
        /// The register needed, according to the gdb numbering scheme.
        register: u16,
        /// The expression object to resume.
        expression: AgentExpressionNeedsRegister<'bc, 's>,
    },
    /// The agent expression needs a memory value to continue evaluation.
    NeedsMemory {
        /// The address at which the value needed lives.
        address: Value,
        /// The size (in bytes) of the memory value needed.
        size: u8,
        /// The expression object to resume.
        expression: AgentExpressionNeedsMemory<'bc, 's>,
    },
    /// The agent expression evaluation is complete and produced a value.
    Complete(Value),
}

impl<'bc, 's> AgentExpressionResult<'bc, 's> {
    /// Unwrap the result, panicking if it is not `Complete`.
    #[allow(clippy::panic)]
    pub fn unwrap(self) -> Value {
        match self {
            AgentExpressionResult::NeedsRegister { .. }
            | AgentExpressionResult::NeedsMemory { .. } => {
                panic!(".unwrap() called, but agent expression evaluation is not complete!")
            }
            AgentExpressionResult::Complete(v) => v,
        }
    }
}

/// Wraps a ManagedSlice in a vec-like interface.
#[derive(Debug)]
struct ManagedVec<'a, T> {
    buf: ManagedSlice<'a, T>,
    len: usize,
}

impl<'a, T> ManagedVec<'a, T> {
    fn new(buf: ManagedSlice<'a, T>) -> Self {
        ManagedVec { buf, len: 0 }
    }

    fn push(&mut self, value: T) -> core::result::Result<(), T> {
        if self.len < self.buf.len() {
            self.buf[self.len] = value;
            self.len += 1;
            Ok(())
        } else {
            match &mut self.buf {
                ManagedSlice::Borrowed(_) => Err(value),
                #[cfg(feature = "alloc")]
                ManagedSlice::Owned(buf) => {
                    self.len += 1;
                    buf.push(value);
                    Ok(())
                }
            }
        }
    }

    fn pop(&mut self) -> Option<T>
    where
        T: Copy,
    {
        if self.len == 0 || self.buf.is_empty() {
            return None;
        }
        self.len -= 1;

        match &mut self.buf {
            ManagedSlice::Borrowed(buf) => Some(buf[self.len]),
            #[cfg(feature = "alloc")]
            ManagedSlice::Owned(buf) => buf.pop(),
        }
    }
}

impl<'a, T> core::ops::Deref for ManagedVec<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.buf
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::panic)]

    use super::*;

    #[test]
    fn test_end() {
        let bytecode = vec![Opcode::OpEnd as u8];
        let result = evaluate(&bytecode);
        assert!(result.is_err());
    }

    #[test]
    fn test_const8() {
        let bytecode = vec![Opcode::OpConst8 as u8, 0x80, Opcode::OpEnd as u8];
        let result = evaluate(&bytecode);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.unwrap(), Value(0x80));
    }

    #[test]
    fn test_const16() {
        let bytecode = vec![Opcode::OpConst16 as u8, 0x80, 0x80, Opcode::OpEnd as u8];
        let result = evaluate(&bytecode);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.unwrap(), Value(0x8080));
    }

    #[test]
    fn test_const32() {
        let bytecode = vec![
            Opcode::OpConst32 as u8,
            0xde,
            0xad,
            0xbe,
            0xef,
            Opcode::OpEnd as u8,
        ];
        let result = evaluate(&bytecode);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.unwrap(), Value(0xdeadbeef));
    }

    #[test]
    fn test_const64() {
        let bytecode = vec![
            Opcode::OpConst64 as u8,
            0x5a,
            0x5a,
            0x5a,
            0x5a,
            0x5a,
            0x5a,
            0x5a,
            0x5a,
            Opcode::OpEnd as u8,
        ];
        let result = evaluate(&bytecode);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.unwrap(), Value(0x5a5a5a5a5a5a5a5a));
    }

    #[test]
    fn test_add() {
        let bytecode = vec![
            Opcode::OpConst8 as u8,
            0x80,
            Opcode::OpConst8 as u8,
            0x80,
            Opcode::OpAdd as u8,
            Opcode::OpEnd as u8,
        ];
        let result = evaluate(&bytecode);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.unwrap(), Value(0x100));
    }

    #[test]
    fn test_add_ovf() {
        let bytecode = vec![
            Opcode::OpConst8 as u8,
            0x0,
            Opcode::OpConst64 as u8,
            0xff,
            0xff,
            0xff,
            0xff,
            0xff,
            0xff,
            0xff,
            0xff,
            Opcode::OpAdd as u8,
            Opcode::OpEnd as u8,
        ];
        let result = evaluate(&bytecode);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.unwrap(), Value(-1));
    }

    #[test]
    fn test_sub() {
        let bytecode = vec![
            Opcode::OpConst8 as u8,
            0x80,
            Opcode::OpConst8 as u8,
            0x7f,
            Opcode::OpSub as u8,
            Opcode::OpEnd as u8,
        ];
        let result = evaluate(&bytecode);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.unwrap(), Value(1));
    }

    #[test]
    fn test_sub_ovf() {
        let bytecode = vec![
            Opcode::OpConst8 as u8,
            0x0,
            Opcode::OpConst64 as u8,
            0xff,
            0xff,
            0xff,
            0xff,
            0xff,
            0xff,
            0xff,
            0xff,
            Opcode::OpSub as u8,
            Opcode::OpEnd as u8,
        ];
        let result = evaluate(&bytecode);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.unwrap(), Value(1));
    }

    #[test]
    fn test_mul() {
        let bytecode = vec![
            Opcode::OpConst8 as u8,
            3,
            Opcode::OpConst8 as u8,
            8,
            Opcode::OpMul as u8,
            Opcode::OpEnd as u8,
        ];
        let result = evaluate(&bytecode);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.unwrap(), Value(24));
    }

    #[test]
    fn test_lsh() {
        let bytecode = vec![
            Opcode::OpConst8 as u8,
            3,
            Opcode::OpConst8 as u8,
            3,
            Opcode::OpLsh as u8,
            Opcode::OpEnd as u8,
        ];
        let result = evaluate(&bytecode);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.unwrap(), Value(24));
    }

    #[test]
    fn test_rsh_unsigned() {
        let bytecode = vec![
            Opcode::OpConst8 as u8,
            24,
            Opcode::OpConst8 as u8,
            3,
            Opcode::OpRshUnsigned as u8,
            Opcode::OpEnd as u8,
        ];
        let result = evaluate(&bytecode);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.unwrap(), Value(3));
    }

    #[test]
    fn test_ext() {
        let bytecode = vec![
            Opcode::OpConst8 as u8,
            0x8,
            Opcode::OpExt as u8,
            0x4,
            Opcode::OpEnd as u8,
        ];
        let result = evaluate(&bytecode);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.unwrap(), Value(-8));
    }

    #[test]
    fn test_rsh_signed() {
        let bytecode = vec![
            Opcode::OpConst8 as u8,
            8,
            Opcode::OpExt as u8,
            4,
            Opcode::OpConst8 as u8,
            1,
            Opcode::OpRshSigned as u8,
            Opcode::OpEnd as u8,
        ];
        let result = evaluate(&bytecode);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.unwrap(), Value(-4));
    }

    #[test]
    fn test_reg() {
        let bytecode = vec![Opcode::OpReg as u8, 0, 16, Opcode::OpEnd as u8];
        let result = evaluate(&bytecode);
        assert!(result.is_ok());
        let result = result.unwrap();
        let result = match result {
            AgentExpressionResult::NeedsRegister {
                register,
                expression,
            } => {
                assert_eq!(register, 16);
                expression.resume_with_register(Value(512))
            }
            _ => panic!(),
        };
        let result = result.unwrap();
        assert_eq!(result.unwrap(), Value(512));
    }

    #[test]
    fn test_mem() {
        let bytecode = vec![
            Opcode::OpConst32 as u8,
            0xde,
            0xad,
            0xbe,
            0xef,
            Opcode::OpRef32 as u8,
            Opcode::OpEnd as u8,
        ];
        let result = evaluate(&bytecode);
        assert!(result.is_ok());
        let result = result.unwrap();
        let result = match result {
            AgentExpressionResult::NeedsMemory {
                address,
                size,
                expression,
            } => {
                assert_eq!(address, Value(0xdeadbeef));
                assert_eq!(size, 4);
                expression.resume_with_memory(Value(512))
            }
            _ => panic!(),
        };
        let result = result.unwrap();
        assert_eq!(result.unwrap(), Value(512));
    }
}
