from typing import Optional, Dict
from pandas import DataFrame, read_hdf
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers.csv_logs import CSVLogger
import pytorch_lightning as pl


from nenequitia.models.baseline import BaselineModule
from nenequitia.codecs import BaselineEncoder
from nenequitia.datasets import BaselineDataFrameDataset
from nenequitia.contrib import get_manuscripts_and_lang_kfolds


def train_baseline_from_hdf5_dataframe(
    train: DataFrame,
    dev: DataFrame,
    test: Optional[DataFrame] = None,
    batch_size: int = 256,
    lr = 1e-4,
    logger_kwargs: Optional[Dict] = None,
    hparams: Optional[Dict] = None
):
    encoder = BaselineEncoder.from_dataframe(train)
    # Hack
    encoder.features = [
        feat[1:]
        for feat in ['$_de', '$nt_', '$es_', '$ent', '$it_', '$_le', '$le_', '$_se', '$et_', '$est', '$re_', '$ne_', '$e_s', '$_es', '$de_', '$te_', '$st_', '$e_d', '$t_a', '$_⁊_', '$_en', '$_si', '$ant', '$t_e', '$e_p', '$_po', '$e_e', '$e_l', '$en_', '$_la', '$ent_', '$t_d', '$_au', '$is_', '$_ne', '$us_', '$_qu', '$_co', '$_de_', '$t_l', '$e_a', '$ie_', '$ut_', '$s_e', '$t_s', '$les', '$se_', '$_me', '$s_a', '$la_', '$_ce', '$_e_', '$_pe', '$_pa', '$ẽ_', '$t_p', '$res', '$s_s', '$s_d', '$_no', '$ns_', '$_li', '$il_', '$_est', '$tre', '$e_c', '$ere', '$oit', '$ont', '$_ma', '$ue_', '$s_p', '$_et', '$_a_', '$_re', '$_sa', '$_so', '$_te', '$_di', '$ien', '$e_m', '$ũ_', '$er_', '$e_t', '$e_n', '$si_', '$li_', '$_mo', '$ant_', '$e_de', '$or_', '$on_', '$t_i', '$_do', '$_s_', '$les_', '$_le_', '$ene', '$_al', '$ist', '$_to', '$ren', '$ui_', '$at_', '$t_m', '$ier', '$des', '$s_l', '$est_', '$nte', '$e_q', '$oit_', '$_su', '$e_f', '$t_c', '$ens', '$ele', '$s_c', '$̃t_', '$ur_', '$ce_', '$_et_', '$_lo', '$̃_s', '$t_de', '$me_', '$t_t', '$_il', '$_ne_', '$e_i', '$ste', '$___', '$ure', '$_la_', '$tes', '$e_u', '$ot_', '$par', '$ont_', '$t_q', '$ten', '$rs_', '$_en_', '$_fe', '$ar_', '$ite', '$_ue', '$ee_', '$as_', '$men', '$os_', '$t_n', '$sen', '$_les', '$_fu', '$_du', '$s_de', '$_ẽ', '$_ie', '$_pr', '$_fa', '$_pl', '$̃_p', '$_des', '$oie', '$s_m', '$_ta', '$son', '$s_n', '$int', '$i_s', '$ire', '$que', '$res_', '$e_se', '$_li_', '$̃_e', '$s_q', '$e_le', '$i_d', '$t_u', '$ta_', '$i_a', '$mes', '$_ad', '$qui', '$_at', '$t_le', '$ore', '$len', '$a_p', '$el_', '$one', '$i_e', '$____', '$e_⁊', '$tie', '$_pu', '$_i_', '$_se_', '$ost', '$ale', '$nes', '$_an', '$_si_', '$uis', '$t._', '$str', '$_ca', '$aut', '$s_i', '$sto', '$uer', '$_est_', '$a_s', '$_q̃', '$_fi', '$por', '$_as', '$ã_', '$ete', '$_ui', '$uit', '$_ch', '$_ar', '$̃_a', '$_cu', '$ate', '$uoi', '$e._', '$_p_', '$s_t', '$_da', '$t_f', '$_par', '$ali', '$_____', '$rent', '$lle', '$uen', '$eur', '$_in', '$_fo', '$ses', '$dis', '$_il_', '$_uo', '$rt_', '$_ho', '$tre_', '$i_p', '$ient', '$eme', '$auo', '$_tu', '$õ_', '$ein', '$t_⁊', '$roi', '$_les_', '$tit', '$pre', '$e_o', '$̃_d', '$nt_a', '$r_l', '$ran', '$ati', '$ĩ_', '$_ap', '$end', '$q̃_', '$sei', '$ter', '$toi', '$a_d', '$tan', '$s_u', '$ort', '$out', '$ons', '$ese', '$ist_', '$our', '$sse', '$ine', '$_mi', '$ẽt', '$tat', '$_tr', '$t_en', '$r_a', '$_qi', '$lie', '$dit', '$nt_l', '$es_d', '$e_r', '$sti', '$a_a', '$eus', '$ez_', '$esto', '$t_o', '$tes_', '$_na', '$ita', '$_ẽ_', '$don', '$ait', '$r_e', '$con', '$_ai', '$a_m', '$t_h', '$_qui', '$_lu', '$a_c', '$_que', '$del', '$lor', '$a_e', '$n_a', '$_ha', '$ert', '$n_e', '$s_f', '$sit', '$are', '$es_e', '$ler', '$_el', '$_eu', '$s_⁊', '$e_po', '$eti', '$es_a', '$ous', '$_on', '$oe_', '$ole', '$ir_', '$_io', '$e_la', '$._s', '$i_c', '$ti_', '$ser', '$ia_', '$ñ_', '$_un', '$ute', '$nt_e', '$ces', '$_q̃_', '$a_l', '$̃_i', '$_am', '$ile', '$ise', '$ere_', '$l_e', '$t_se', '$den', '$ais', '$i_l', '$in_', '$s_o', '$ill', '$ues', '$r_d', '$e_si', '$stre', '$e_⁊_', '$_ae', '$_he', '$cor', '$man', '$ois', '$n_d', '$este', '$n_p', '$n_l', '$_ou', '$ors', '$lt_', '$tis', '$ust', '$sa_', '$_ac', '$e_ne', '$rent_', '$nen', '$_mn', '$nos', '$aue', '$̃_c', '$san', '$oi_', '$eue', '$s._', '$rie', '$uie', '$art', '$_ĩ', '$leu', '$e_h', '$es_p', '$_nu', '$._e', '$_por', '$ame', '$_or', '$__e', '$nde', '$_ro', '$tet', '$_del', '$ele_', '$ies', '$_q_', '$ane', '$tou', '$ũt', '$n_s', '$_mu', '$e_es', '$die', '$ens_', '$r_s', '$e_de_', '$int_', '$ome', '$l_a', '$ret', '$_son', '$a_t', '$iti', '$_st', '$sie', '$ai_', '$che', '$lui', '$nt_d', '$deu', '$_d_', '$lit', '$que_', '$eli', '$_vo', '$t_⁊_', '$ala', '$eu_', '$s_le', '$_ut', '$eri', '$_er', '$ient_', '$mes_', '$e_en', '$_ce_', '$_ee', '$com', '$era', '$_sen', '$_̃_', '$non', '$e_v', '$⁊_s', '$ign', '$eni', '$t_es', '$auoi', '$̃_n', '$ãt', '$esp', '$iss', '$pen', '$tũ', '$cel', '$tur', '$_ci', '$per', '$eis', '$sta', '$ili', '$ra_', '$nz_', '$aus', '$ente', '$t_la', '$enc', '$un_', '$_qe', '$it_a', '$stoi', '$oir', '$a_n', '$t_r', '$_ti', '$ment', '$l_s', '$lan', '$eno', '$__s', '$une', '$i_m', '$r_c', '$ses_', '$ou_', '$_ia', '$_sei', '$nce', '$nt_s', '$_dis', '$uit_', '$aui', '$_l_', '$ers', '$nti', '$_._', '$e_co', '$ain', '$i_n', '$e_me', '$cha', '$̃_u', '$oien', '$estoi', '$e_so', '$mor', '$rte', '$det', '$ata', '$_n_', '$_hu', '$ans', '$̃_t', '$to_', '$e_ce', '$tor', '$̃_m', '$a_u', '$e_sa', '$no_', '$n_c', '$e_qu', '$̃s_', '$mie', '$_gr', '$e_pe', '$ris', '$ae_', '$tt_', '$i_t', '$ple', '$t_de_', '$s_se', '$tin', '$_esto', '$es_s', '$mon', '$let', '$io_', '$s_es', '$mer', '$es_l', '$enu', '$na_', '$and', '$ena', '$tant', '$ner', '$_ie_', '$des_', '$net', '$ors_', '$eit', '$i_i', '$tẽ', '$_t_', '$ble', '$t_au', '$seu', '$ast', '$pos', '$oient', '$es_c', '$lon', '$s̃_', '$ei_', '$qui_', '$_pt', '$o_e', '$_u_', '$nte_', '$a_f', '$ose', '$rai', '$_ve', '$au_', '$s_en', '$t̃_', '$_pi', '$ste_', '$e_b', '$sein', '$ons_', '$out_', '$_auo', '$_ꝑ_', '$s_⁊_', '$nt_p', '$t_li', '$son_', '$t_il', '$ua_', '$ad_', '$t_si', '$s_h', '$rer', '$par_', '$_ei', '$e_g', '$gra', '$de_s', '$i_f', '$de_l', '$om_', '$ene_', '$._a', '$es_de', '$_it', '$esc', '$it_e', '$nes_', '$ieu', '$uan', '$o_d', '$ule', '$esi', '$re_s', '$_bi', '$_ñ', '$eut', '$e_pa', '$ell', '$qe_', '$dem', '$bie', '$pon', '$ien_', '$o_s', '$_don', '$ote', '$ous_', '$tres', '$_⁊_s', '$ne_s', '$ess', '$._d', '$ins', '$cou', '$e_te', '$car', '$_deu', '$_is', '$dist', '$r_p', '$mne', '$ire_', '$set', '$tel', '$_es_', '$atu', '$_ent', '$t_po', '$pou', '$tot', '$ite_', '$ssi', '$i._', '$_be', '$_pre', '$ion', '$al_', '$anc', '$i_de', '$nor', '$_con', '$mai', '$fai', '$_ge', '$_ps', '$oue', '$i_q', '$e_no', '$qua', '$re.', '$ton', '$qi_', '$_com', '$lis', '$_cr', '$_bo', '$⁊_d', '$e_li', '$le_p', '$itu', '$lus', '$nt_m', '$i_u', '$di_', '$fer', '$lau', '$uis_', '$rre', '$sent', '$st_a', '$e_au', '$_dt', '$_ra', '$._q', '$ace', '$_que_', '$soi', '$mar', '$sic', '$pe_', '$l_l', '$_ses', '$et_s', '$t_qu', '$toit', '$for', '$d_e', '$_c_', '$eo_', '$_ꝙ_', '$uet', '$te_d', '$t_a_', '$der', '$eso', '$_s_s', '$._⁊', '$_me_', '$_mes', '$e_et', '$eta', '$es_m', '$iet', '$chi', '$ait_', '$et_d', '$tio', '$one_', '$tra', '$le_s', '$lle_', '$nõ', '$_p̃', '$et_e', '$plu', '$s_de_', '$_ao', '$nit', '$it_d', '$nt_i', '$isi', '$s_r', '$⁊_a', '$e_mo', '$nt.', '$̃_q', '$s_au', '$noi', '$rti', '$et_a', '$loi', '$emen', '$ela', '$a_i', '$ass', '$ero', '$ron', '$edi', '$s_ne', '$cie', '$e_ma', '$lat', '$l_p', '$eua', '$rem', '$e_di', '$tu_', '$o_a', '$nat', '$uo_', '$iere', '$o_p', '$nie', '$ea_', '$t_pa', '$qil', '$_oi', '$_nos', '$es.', '$ier_', '$unt', '$uoit', '$⁊_p', '$t_le_', '$r_n', '$̃_l', '$le_d', '$u_d', '$rant', '$r_m', '$_qui_', '$_eo', '$tas', '$ent_a', '$ndi', '$cit', '$dre', '$⁊_e', '$cen', '$es_q', '$t_b', '$nse', '$ure_', '$_len', '$e_e_', '$tũ_', '$hie', '$_cor', '$uin', '$ne_p', '$_tou', '$_fr', '$ore_', '$_de_l', '$stre_', '$it.', '$s_s_', '$reu', '$̃e_', '$e_do', '$e_re', '$ide', '$n_m', '$ũt_', '$_pen', '$pro', '$̃_f', '$ett', '$_te_', '$mal', '$_par_', '$_s_p', '$r_t', '$_ex', '$re_e', '$pet', '$s_po', '$e_est', '$_ces', '$e_ne_', '$r_i', '$u_p', '$stu', '$s\uf1ac_', '$t_al', '$ne_d', '$_ũ', '$ede', '$tte', '$estr', '$_den', '$n_t', '$em_', '$sem', '$__p', '$r_le', '$s_qu', '$pes', '$_ba', '$tã', '$r._', '$use', '$sit_', '$_s̃', '$nt_c', '$ul_', '$_aut', '$__a', '$r_de', '$i_se', '$e_le_', '$ice', '$l_d', '$tut', '$nt_q', '$cũ', '$t_co', '$eren', '$aut_', '$sil', '$_sa_', '$_hi', '$l_n', '$et_l', '$re_d', '$e_lo', '$dit_', '$̃._', '$es_n', '$uel', '$an_', '$ato', '$_tre', '$en_l', '$u_a', '$nta', '$ge_', '$t_ce', '$_ea', '$it_s', '$ees', '$gne', '$_ad_', '$hom', '$t_il_', '$_ser', '$por_', '$_ã', '$eat', '$_ua', '$aur', '$e_to', '$_de_s', '$_ab', '$ers_', '$_res', '$_qe_', '$idi', '$met', '$ans_', '$_lor', '$_fai', '$err', '$uil', '$nẽ', '$nou', '$ust_', '$u_s', '$eil', '$eru', '$s_e_', '$_ot', '$lai', '$u_e', '$r_q', '$de_c', '$iez', '$eur_', '$stoit', '$gen', '$alu', '$sis', '$ior', '$_e_s', '$e_la_', '$tia', '$te_s', '$roit', '$_ĩ_', '$ẽt_', '$aie', '$toit_', '$ece', '$t_g', '$ase', '$lo_', '$_mar', '$nis', '$_sein', '$ne_l', '$ade', '$iẽ', '$pat', '$e_a_', '$_nt', '$nt_de', '$_⁊_d', '$it_i', '$ura', '$ntr', '$lem', '$lia', '$tro', '$sor', '$_ni', '$_pos', '$_gra', '$lor_', '$_ten', '$pas', '$rit', '$_dist', '$l_m', '$tit_', '$st_e', '$_cl', '$etu', '$_det', '$n_de', '$te_e', '$ent_l', '$tis_', '$t_ma', '$t_me', '$pai', '$_ali', '$lent', '$._c', '$rec', '$_iu', '$ian', '$_auoi', '$t_ne', '$_ut_', '$eint', '$sau', '$_pou', '$_nõ', '$lee', '$_⁊_p', '$pot', '$s_co', '$lus_', '$re_l', '$eul', '$cre', '$_os', '$re_a', '$dou', '$_ens', '$our_', '$t_pe', '$_sit', '$_cha', '$ss_', '$._i', '$rou', '$a_de', '$et_p', '$ari', '$p_s', '$dic', '$uant', '$_tan', '$lli', '$_uu', '$nt_le', '$t_sa', '$st_d', '$coi', '$r_u', '$̃_⁊', '$rest', '$_for', '$uoit_', '$poe', '$ni_', '$lam', '$ement', '$le_c', '$it_l', '$_san', '$_plu', '$ort_', '$rei', '$tus', '$poi', '$_ale', '$a_q', '$uti', '$tant_', '$mei', '$dont', '$_cel', '$_s̃_', '$n_n', '$ond', '$s_si', '$._et', '$sin', '$_qil', '$_oe', '$bien', '$le_m', '$t᷑_', '$t_re', '$esa', '$_ga', '$_leu', '$eus_', '$ici', '$e_su', '$ille', '$ies_', '$mis', '$oen', '$nt_t', '$aul', '$ail', '$cer', '$u_l', '$ete_', '$̃te', '$pue', '$_mon', '$t_so', '$_sic', '$ult', '$le_e', '$sle', '$_ss', '$_em', '$am_', '$_men', '$qil_', '$ori', '$en_a', '$ree', '$rat', '$_qi_', '$_cou', '$t_en_', '$age', '$s_me', '$pie', '$ssa', '$amo', '$ano', '$m̃_', '$gran', '$quo', '$st_l', '$oir_', '$une_', '$_qͥ', '$mou', '$sig', '$rẽ', '$nõ_', '$utr', '$⁊_l', '$fu_', '$ostr', '$lla', '$uue', '$uta', '$e_et_', '$_hom', '$chie', '$es_t', '$lui_', '$ceu', '$dt_', '$tout', '$eto', '$t_te', '$e_ie', '$ual', '$nan', '$ard', '$moi', '$o_n', '$ert_', '$da_', '$_mal', '$_aa', '$_ml', '$ma_', '$eui', '$_vi', '$pla', '$ani', '$s_g', '$n_f', '$_dit', '$nsi', '$auoit', '$ne_e', '$_des_', '$ois_', '$pres', '$cil', '$gno', '$tent', '$he_', '$ĩt', '$_dn', '$_son_', '$elle', '$lar', '$̃_si', '$do_', '$t_et', '$sũ', '$it_p', '$_bie', '$es_f', '$_dem', '$sus', '$um_', '$ues_', '$s_pe', '$_qua', '$_fer', '$s_est', '$uir', '$inz', '$alo', '$ps_', '$tui', '$̃_de', '$sou', '$us_e', '$t_to', '$_au_', '$_⁊_a', '$__s_', '$_dr', '$t_est', '$i_o', '$pri', '$_ñ_', '$l_c', '$oli', '$air', '$ema', '$eure', '$cest', '$ate_', '$._⁊_', '$eit_', '$enr', '$t_li_', '$ent_e', '$ute_', '$m_e', '$cõ', '$eau', '$s_no', '$_o_', '$_p_s', '$uz_', '$nos_', '$_sl', '$_ala', '$ne_a', '$emi', '$ape', '$e_fa', '$ex_', '$_gu', '$_ul', '$i_r', '$tem', '$tai', '$onte', '$_seu', '$_per', '$este_', '$̃ti', '$esti', '$erent', '$t_e_', '$teu', '$_uen', '$adi', '$t_no', '$_mor', '$te_p', '$iso', '$tue', '$e_s_', '$ut_a', '$s_et', '$i_co', '$ts_', '$ñt', '$sio', '$de_p', '$e_fu', '$cle', '$urs', '$s_pa', '$iat', '$t_mo', '$spe', '$ine_', '$nee', '$_qr', '$nel', '$nui', '$la_p', '$pui', '$_mai', '$ls_', '$t_do', '$st_s', '$te_a', '$ara', '$z_d', '$not', '$._et_', '$rme', '$olo', '$_pon', '$dont_', '$ora', '$tie_', '$dist_', '$e_fe', '$s_la', '$sol', '$_aue', '$sam', '$t_les', '$dan', '$m_a', '$is_a', '$sir', '$ostre', '$s_b', '$ẽ_s', '$le_l', '$ais_', '$a._', '$_die', '$aua', '$uli', '$_fu_', '$s_p_', '$see', '$cont', '$us_a', '$_sp', '$_at_', '$eon', '$eie', '$u_m', '$o_m', '$tee', '$nr_', '$r_⁊', '$de_m', '$s_li', '$uon', '$q\uf1ac_', '$ent.', '$uei', '$_de_c', '$i_es', '$ndr', '$es_⁊', '$o_i', '$is_e', '$̃_h', '$tos', '$n_q', '$._p', '$s_a_', '$_nes', '$_pro', '$dec', '$inz_', '$p̃_', '$ut_e', '$iũ', '$eut_', '$cat', '$_pot', '$ueu', '$en_s', '$uol', '$es_i', '$car_', '$st_p', '$t_la_', '$es_o', '$_om', '$oes', '$sai', '$enl', '$ie.', '$tẽ_', '$sent_', '$l_i', '$._l', '$_ed', '$gue', '$us_d', '$_man', '$_ele', '$arti', '$a_r', '$aro', '$tõ', '$le_a', '$re_p', '$liu', '$dea', '$_poe', '$udi', '$ile_', '$ment_', '$oti', '$alt', '$es_u', '$_e_p', '$̃_⁊_', '$de_d', '$ns_e', '$nũ', '$t_v', '$d_i', '$rde', '$e_des', '$u_n', '$elu', '$o_l', '$⁊_i', '$n_u', '$een', '$i_le', '$oni', '$gie', '$is_s', '$ins_', '$_⁊_e', '$t_di', '$_qn', '$bre', '$att', '$emo', '$qͥ_', '$̃_s_', '$ent_d', '$e_pr', '$mat', '$o_t', '$bon', '$_dou', '$_bien', '$las', '$_tot', '$nus', '$_s\uf1ac', '$cũ_', '$t_lo', '$uent', '$nt_n', '$e_du', '$rel', '$aci', '$bien_', '$eres', '$ãt_', '$dest', '$ibi', '$seint', '$_ses_', '$neu', '$ona', '$_nõ_', '$_le_p', '$plus', '$li_d', '$_us', '$sse_', '$s_ce', '$se_d', '$s_di', '$aus_', '$sen_', '$_ds', '$ioi', '$enti', '$uss', '$ost_', '$nost', '$cho', '$ie_d', '$rant_', '$_che', '$e_uo', '$ime', '$dat', '$gar', '$en_e', '$ean', '$ie_e', '$iue', '$u_c', '$aen', '$_por_', '$te_l', '$tal', '$ius', '$tet_', '$iui', '$c_s', '$ono', '$__l', '$esu', '$ito', '$_pn', '$nt_en', '$e_les', '$ntre', '$ila', '$̃_ẽ', '$_sig', '$iz_', '$einz', '$ic_', '$ance', '$r_la', '$_qil_', '$sat', '$nent', '$._si', '$.⁊_', '$r_o', '$i_⁊', '$n_o', '$_lau', '$s_ma', '$_tn', '$estre', '$t_as', '$_ant', '$roit_', '$nda', '$aco', '$utre', '$te.', '$_s\uf1ac_', '$bi_', '$_mie', '$e_al', '$_non', '$apr', '$le_f', '$abi', '$doi', '$_sie', '$_tant', '$e_li_', '$_ũ_', '$it._', '$e_en_', '$_ip', '$it_c', '$sui', '$c̃_', '$inte', '$l_u', '$_⁊_l', '$_pas', '$en_p', '$ica', '$d_a', '$⁊_c', '$⁊_t', '$us_s', '$ant_l', '$la_c', '$de_t', '$uat', '$z_e', '$e_fo', '$_dl', '$fie', '$dir', '$pour', '$l_es', '$n_i', '$e_ta', '$o_c', '$_aui', '$tri', '$du_', '$uant_', '$lic', '$pel', '$nue', '$ula', '$cet', '$l_f', '$oui', '$elo', '$eet', '$nco', '$ce_q', '$ina', '$_quo', '$ne_f', '$_ol', '$cur', '$uou', '$lei', '$nto', '$cele', '$mẽ', '$rue', '$lue', '$n_la', '$mi_', '$asi', '$it_m', '$po_', '$s_so', '$e_se_', '$_lon', '$_pe_', '$pt_', '$ns_a', '$is.', '$se_s', '$il_e', '$onc', '$er_e', '$ns_d', '$\uf1ac_d', '$d_s', '$due', '$__o', '$ut_d', '$et_q', '$a_le', '$_sc', '$e_ue', '$olu', '$s_re', '$mno', '$i_en', '$ca_', '$peu', '$ter_', '$nne', '$_ur', '$_pla', '$e\uf1ac_', '$se_p', '$rne', '$dee', '$_tes', '$oie_', '$uos', '$_ne_s', '$ũ_e', '$o_u', '$p_p', '$rep', '$c_e', '$it_n', '$_va', '$ante', '$nten', '$unt_', '$l_q', '$sont', '$e_ch', '$a_g', '$n_es', '$_as_', '$_pes', '$pit', '$nst', '$ense', '$_e_e', '$til', '$_ont', '$ie_s', '$vou', '$ira', '$port', '$is_d', '$n_le', '$e_pu', '$uid', '$fait', '$le_t', '$_sem', '$ini', '$fus', '$leur', '$uui', '$igno', '$dam', '$er_a', '$asse', '$nau', '$a_se', '$lac', '$_s_s_', '$let_', '$nt_f', '$einz_', '$⁊_m', '$ume', '$tend', '$_sol', '$_set', '$sso', '$_cest', '$s_ne_', '$il_a', '$tat_', '$lẽ', '$_pet', '$et_t', '$_mer', '$z_a', '$_lor_', '$pli', '$_pat', '$iau', '$eles', '$_gran', '$emp', '$_c̃', '$ane_', '$ie_a', '$pee', '$\uf1ac_s', '$rce', '$ome_', '$u_t', '$_tout', '$_s_c', '$oul', '$nt_⁊', '$it_de', '$_nou', '$_de_m', '$e_tr', '$_s_e', '$eis_', '$tau', '$mbl', '$pte', '$orte', '$e_par', '$_sit_', '$_mei', '$r_en', '$r_f', '$tar', '$isse', '$\uf1ac_a', '$tre_s', '$are_', '$eci', '$sce', '$_ple', '$a_po', '$esta', '$il_n', '$ũ_s', '$ble_', '$_ue_', '$ces_', '$ut_s', '$a_o', '$__e_', '$rui', '$ang', '$han', '$uce', '$rez', '$_gn', '$oll', '$s_do', '$i_g', '$ne_m', '$aes', '$e_fi', '$lio', '$ẽ_p', '$nul', '$rau', '$_br', '$e_tu', '$qr_', '$_poi', '$mos', '$_ou_', '$oet', '$iez_', '$sme', '$bit', '$_sau', '$tei', '$nest', '$r_ce', '$usi', '$l_de', '$oin', '$erre', '$_pue', '$_dec', '$ci_', '$\uf1ac_e', '$enn', '$uns', '$_car', '$_qͥ_', '$t_ad', '$dẽ', '$cui', '$e_qi', '$etr', '$̃_̃', '$_m_', '$dũ', '$col', '$plus_', '$_ri', '$ese_', '$̃_o', '$autr', '$pere', '$ent_p', '$rea', '$set_', '$̃_r', '$nt_u', '$e_pl', '$all', '$a_me', '$sẽ', '$anu', '$en_d', '$b\uf1ac_', '$tus_', '$esl', '$tea', '$_sou', '$so_', '$d_d', '$et_i', '$l_se', '$ures', '$l_ne', '$dei', '$bes', '$a_b', '$ret_', '$_a_l', '$iens', '$it_u', '$ras', '$tres_', '$et_c', '$_dest', '$s_v', '$er.', '$t_.', '$_sũ', '$st_m', '$spo', '$e_vo', '$sur', '$s_mo', '$_plus', '$sot', '$lte', '$sant', '$amb', '$_esp', '$le_n', '$ri_', '$uoir', '$seinz', '$sue', '$_de_p', '$dui', '$nre', '$lant', '$õt', '$nsa', '$nce_', '$ẽs', '$e_da', '$uet_', '$ule_', '$can', '$gre', '$_ee_', '$net_', '$s_et_', '$ec_', '$_d̃', '$tate', '$tas_', '$t_su', '$abl', '$orr', '$iou', '$tot_', '$eint_', '$t_at', '$mie_', '$lil', '$_un_', '$et_de', '$it_t', '$ro_', '$soit', '$bat', '$dut', '$z_l', '$e_.', '$re_q', '$m_i', '$icu', '$_lui', '$de_la', '$be_', '$ale_', '$etit', '$ded', '$i_di', '$s_le_', '$d_p', '$t_pl', '$_ren', '$min', '$har', '$t_ho', '$ala_', '$_tt', '$_p̃_', '$_cũ', '$i_h', '$ana', '$isa', '$qu_', '$rac', '$toie', '$⁊_de', '$_en_l', '$ne_n', '$ne_de', '$_id', '$ut_l', '$ntu', '$igne', '$es_le', '$ama', '$ent_s', '$anz', '$_e_d', '$._n', '$_ps_', '$ut_p', '$._m', '$nẽ_', '$fen', '$los', '$te_de', '$_sil', '$tiu', '$̃_se', '$s_sa', '$est_a', '$nt_h', '$o_q', '$uor', '$_tro', '$s_te', '$le_de', '$.si', '$te_n', '$l_t', '$_a_s', '$cue', '$i_est', '$tout_', '$lib', '$it_q', '$fre']
    ]

    if not hparams:
        hparams = {}

    # data
    train_loader = DataLoader(
        BaselineDataFrameDataset(train, encoder), batch_size=batch_size, collate_fn=encoder.collate_gt,
        num_workers=4,
        shuffle=True
    )
    val_loader = DataLoader(
        BaselineDataFrameDataset(dev, encoder), batch_size=batch_size, collate_fn=encoder.collate_gt,
        num_workers=4
    )

    # model
    model = BaselineModule(encoder=encoder, training=True, lr=lr)
    model = BaselineModule(encoder=encoder, training=True, lr=lr)

    # training
    checkpoint_callback = ModelCheckpoint(
        monitor="Dev[Rec]",
        filename="sample-{epoch:02d}",
        save_top_k=1,
        mode="max",
        verbose=True
    )
    logger = CSVLogger(**logger_kwargs)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=16,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='epoch'),
            EarlyStopping(monitor="Dev[Rec]", mode="max", min_delta=2e-3, patience=10, verbose=True)
        ],
        max_epochs=100,
        logger=logger
    )
    model.save_hyperparameters()
    trainer.fit(model, train_loader, val_loader)
    if test is not None:
        test_loader = DataLoader(
            BaselineDataFrameDataset(test, encoder), batch_size=batch_size, collate_fn=encoder.collate_gt,
            num_workers=4
        )
        return model, trainer.test(dataloaders=test_loader)
    return model


if __name__ == "__main__":
    df = read_hdf("texts.hdf5", key="df", index_col=0)

    df["bin"] = ""
    df.loc[df.CER < 10, "bin"] = "Good"
    df.loc[df.CER.between(10, 25, inclusive="left"), "bin"] = "Acceptable"
    df.loc[df.CER.between(25, 50, inclusive="left"), "bin"] = "Bad"
    df.loc[df.CER >= 50, "bin"] = "Very bad"

    for (lr, dropout) in [(5e-4, .1)]:
        for i in range(5):
            train, dev, test = get_manuscripts_and_lang_kfolds(
                df,
                k=i, per_k=2,
                force_test=["SBB_PK_Hdschr25"]
            )
            print("Train dataset description")
            print(train.groupby("bin").size())
            print("Dev dataset description")
            print(dev.groupby("bin").size())
            print("Test dataset description")
            print(test.groupby("bin").size())
            model = train_baseline_from_hdf5_dataframe(
                train, dev, test=test,
                lr=lr, batch_size=128,
                logger_kwargs=dict(
                    save_dir="explogs-baseline",
                    name=f"Baseline",
                    version=str(i)
                )
            )
