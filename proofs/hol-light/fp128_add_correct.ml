(*
 * Functional correctness proof for Akita's AArch64 Fp128 addition kernel.
 *
 * The object path is supplied by the build that produced fp128_add.o. The
 * explicit instruction list makes the theorem fail if those object bytes
 * change.
 *)

needs "arm/proofs/base.ml";;

let akita_fp128_add_object = Sys.getenv "AKITA_FP128_ADD_OBJECT";;

let akita_fp128_add_mc =
  define_assert_from_elf "akita_fp128_add_mc" akita_fp128_add_object
  [
    0xab020005;
    0xba030026;
    0x1a9f37e7;
    0xab0400a8;
    0xba1f00c9;
    0x7a4038e0;
    0x9a851100;
    0x9a861121;
    0xd65f03c0
  ];;

let AKITA_FP128_ADD_EXEC = ARM_MK_EXEC_RULE akita_fp128_add_mc;;

let akita_fp128_a7f7_p = new_definition
 `akita_fp128_a7f7_p = 0xffffffffffffffffffffffff00005809`;;

let AKITA_FP128_ADD_CORRECT = time prove
 (`!a0 a1 b0 b1 pc.
        ensures arm
          (\s. aligned_bytes_loaded s (word pc) akita_fp128_add_mc /\
               read PC s = word pc /\
               read X0 s = a0 /\
               read X1 s = a1 /\
               read X2 s = b0 /\
               read X3 s = b1 /\
               read X4 s = word 0xffffa7f7)
          (\s. read PC s = word (pc + 0x20) /\
               (bignum_of_wordlist [a0; a1] < akita_fp128_a7f7_p /\
                bignum_of_wordlist [b0; b1] < akita_fp128_a7f7_p
                ==> bignum_of_wordlist [read X0 s; read X1 s] =
                    (bignum_of_wordlist [a0; a1] +
                     bignum_of_wordlist [b0; b1]) MOD
                    akita_fp128_a7f7_p))
          (MAYCHANGE [PC; X0; X1; X5; X6; X7; X8; X9] ,,
           MAYCHANGE SOME_FLAGS ,, MAYCHANGE [events])`,
  MAP_EVERY X_GEN_TAC
   [`a0:int64`; `a1:int64`; `b0:int64`; `b1:int64`; `pc:num`] THEN
  REWRITE_TAC[SOME_FLAGS] THEN
  ABBREV_TAC `m = bignum_of_wordlist [a0; a1]` THEN
  ABBREV_TAC `n = bignum_of_wordlist [b0; b1]` THEN
  ENSURES_INIT_TAC "s0" THEN
  ARM_ACCSTEPS_TAC AKITA_FP128_ADD_EXEC [1;2;4;5] (1--8) THEN
  ENSURES_FINAL_STATE_TAC THEN ASM_REWRITE_TAC[] THEN STRIP_TAC THEN
  ABBREV_TAC `l = bignum_of_wordlist [sum_s1; sum_s2]` THEN
  ABBREV_TAC `t = bignum_of_wordlist [sum_s4; sum_s5]` THEN

  (* The first carry chain computes m + n modulo 2^128. *)
  SUBGOAL_THEN `2 EXP 128 * bitval carry_s2 + l = m + n` ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["l"; "m"; "n"] THEN
    REWRITE_TAC[bignum_of_wordlist; MULT_CLAUSES; ADD_CLAUSES] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN REAL_ARITH_TAC;
    ALL_TAC] THEN

  (* The second chain computes l + (2^128 - p) modulo 2^128. *)
  SUBGOAL_THEN `2 EXP 128 * bitval carry_s5 + t = l + 4294944759`
  ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["t"; "l"] THEN
    REWRITE_TAC[bignum_of_wordlist; MULT_CLAUSES; ADD_CLAUSES] THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_CLAUSES] THEN
    ACCUMULATOR_ASSUM_LIST(MP_TAC o end_itlist CONJ o DECARRY_RULE) THEN
    DISCH_THEN(fun th -> REWRITE_TAC[th]) THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `l < 2 EXP 128 /\ t < 2 EXP 128` STRIP_ASSUME_TAC THENL
   [MAP_EVERY EXPAND_TAC ["l"; "t"] THEN BOUNDER_TAC[];
    ALL_TAC] THEN
  DISCARD_STATE_TAC "s8" THEN
  ACCUMULATOR_POP_ASSUM_LIST(K ALL_TAC) THEN

  (* The conditional compare selects t exactly when reduction is needed. *)
  (ASM_CASES_TAC `carry_s2:bool` THENL
    [RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `carry_s2:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `carry_s2:bool`);
     RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `~carry_s2:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~carry_s2:bool`)]) THEN
  (ASM_CASES_TAC `carry_s5:bool` THENL
    [RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `carry_s5:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `carry_s5:bool`);
     RULE_ASSUM_TAC(REWRITE_RULE[ASSUME `~carry_s5:bool`; BITVAL_CLAUSES]) THEN
     ASSUME_TAC(ASSUME `~carry_s5:bool`)]) THEN
  ASM_CASES_TAC `akita_fp128_a7f7_p <= m + n` THEN
  ASM_REWRITE_TAC
   [WORD_SUB_0; VAL_WORD_BITVAL; BITVAL_EQ_0; BITVAL_CLAUSES;
    akita_fp128_a7f7_p; MOD_ADD_CASES; GSYM NOT_LE; COND_SWAP] THEN
  CONV_TAC WORD_REDUCE_CONV THEN ASM_REWRITE_TAC[] THEN
  RULE_ASSUM_TAC(REWRITE_RULE[akita_fp128_a7f7_p; BITVAL_CLAUSES]) THEN
  RULE_ASSUM_TAC(CONV_RULE NUM_REDUCE_CONV) THEN
  ASM_SIMP_TAC[MOD_ADD_CASES; akita_fp128_a7f7_p; GSYM NOT_LE; COND_SWAP] THEN
  CONV_TAC NUM_REDUCE_CONV THEN
  POP_ASSUM_LIST(MP_TAC o end_itlist CONJ) THEN ARITH_TAC);;

let AKITA_FP128_ADD_SUBROUTINE_CORRECT = time prove
 (`!a0 a1 b0 b1 pc returnaddress.
        ensures arm
          (\s. aligned_bytes_loaded s (word pc) akita_fp128_add_mc /\
               read PC s = word pc /\
               read X30 s = returnaddress /\
               read X0 s = a0 /\
               read X1 s = a1 /\
               read X2 s = b0 /\
               read X3 s = b1 /\
               read X4 s = word 0xffffa7f7)
          (\s. read PC s = returnaddress /\
               (bignum_of_wordlist [a0; a1] < akita_fp128_a7f7_p /\
                bignum_of_wordlist [b0; b1] < akita_fp128_a7f7_p
                ==> bignum_of_wordlist [read X0 s; read X1 s] =
                    (bignum_of_wordlist [a0; a1] +
                     bignum_of_wordlist [b0; b1]) MOD
                    akita_fp128_a7f7_p))
          (MAYCHANGE_REGS_AND_FLAGS_PERMITTED_BY_ABI)`,
  ARM_ADD_RETURN_NOSTACK_TAC
    AKITA_FP128_ADD_EXEC AKITA_FP128_ADD_CORRECT);;
