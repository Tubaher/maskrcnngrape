--
-- PostgreSQL database dump
--

-- Dumped from database version 10.10 (Ubuntu 10.10-0ubuntu0.18.04.1)
-- Dumped by pg_dump version 10.10 (Ubuntu 10.10-0ubuntu0.18.04.1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: pg_test; Type: TABLE; Schema: public; Owner: digevo
--

CREATE TABLE public.pg_test (
    id integer NOT NULL,
    client_id integer NOT NULL,
    campo_id integer NOT NULL,
    cultivo_id integer NOT NULL,
    hilera_id integer NOT NULL,
    racimo_id integer NOT NULL,
    racimo_area integer NOT NULL,
    racimo_lng double precision,
    racimo_lat double precision,
    racimo_sepa character varying(50),
    "timestamp" timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


ALTER TABLE public.pg_test OWNER TO digevo;

--
-- Name: pg_test_id_seq; Type: SEQUENCE; Schema: public; Owner: digevo
--

CREATE SEQUENCE public.pg_test_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.pg_test_id_seq OWNER TO digevo;

--
-- Name: pg_test_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: digevo
--

ALTER SEQUENCE public.pg_test_id_seq OWNED BY public.pg_test.id;


--
-- Name: pg_test id; Type: DEFAULT; Schema: public; Owner: digevo
--

ALTER TABLE ONLY public.pg_test ALTER COLUMN id SET DEFAULT nextval('public.pg_test_id_seq'::regclass);


--
-- Data for Name: pg_test; Type: TABLE DATA; Schema: public; Owner: digevo
--

COPY public.pg_test (id, client_id, campo_id, cultivo_id, hilera_id, racimo_id, racimo_area, racimo_lng, racimo_lat, racimo_sepa, "timestamp") FROM stdin;
\.


--
-- Name: pg_test_id_seq; Type: SEQUENCE SET; Schema: public; Owner: digevo
--

SELECT pg_catalog.setval('public.pg_test_id_seq', 1, false);


--
-- Name: pg_test pg_test_pkey; Type: CONSTRAINT; Schema: public; Owner: digevo
--

ALTER TABLE ONLY public.pg_test
    ADD CONSTRAINT pg_test_pkey PRIMARY KEY (id);


--
-- PostgreSQL database dump complete
--

