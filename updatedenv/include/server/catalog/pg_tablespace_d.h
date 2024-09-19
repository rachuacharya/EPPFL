/*-------------------------------------------------------------------------
 *
 * pg_tablespace_d.h
 *    Macro definitions for pg_tablespace
 *
 * Portions Copyright (c) 1996-2019, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 * NOTES
 *  ******************************
 *  *** DO NOT EDIT THIS FILE! ***
 *  ******************************
 *
 *  It has been GENERATED by src/backend/catalog/genbki.pl
 *
 *-------------------------------------------------------------------------
 */
#ifndef PG_TABLESPACE_D_H
#define PG_TABLESPACE_D_H

#define TableSpaceRelationId 1213

#define Anum_pg_tablespace_oid 1
#define Anum_pg_tablespace_spcname 2
#define Anum_pg_tablespace_spcowner 3
#define Anum_pg_tablespace_spcacl 4
#define Anum_pg_tablespace_spcoptions 5

#define Natts_pg_tablespace 5

#define DEFAULTTABLESPACE_OID 1663
#define GLOBALTABLESPACE_OID 1664

#endif							/* PG_TABLESPACE_D_H */
