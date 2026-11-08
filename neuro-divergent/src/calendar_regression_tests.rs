use super::*;

fn ny() -> DataType {
    DataType::Datetime(TimeUnit::Milliseconds, Some("America/New_York".into()))
}
fn instant(value: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(value)
        .unwrap()
        .with_timezone(&Utc)
}

#[test]
fn zoned_month_end_calendar_keeps_midnight_through_dst() {
    let anchor = instant("2026-01-31T05:00:00Z");
    assert_eq!(
        advance_zoned(anchor, &Frequency::Monthly, 1, &ny()).unwrap(),
        instant("2026-02-28T05:00:00Z")
    );
    assert_eq!(
        advance_zoned(anchor, &Frequency::Monthly, 2, &ny()).unwrap(),
        instant("2026-03-31T04:00:00Z")
    );
    assert_eq!(
        advance_zoned(anchor, &Frequency::Quarterly, 1, &ny()).unwrap(),
        instant("2026-04-30T04:00:00Z")
    );
    assert_eq!(
        advance_zoned(anchor, &Frequency::Custom("2M".into()), 1, &ny()).unwrap(),
        instant("2026-03-31T04:00:00Z")
    );
}

#[test]
fn zoned_daily_weekly_and_business_offsets_use_local_calendar_days() {
    let anchor = instant("2026-03-07T05:00:00Z");
    assert_eq!(
        advance_zoned(anchor, &Frequency::Daily, 2, &ny()).unwrap(),
        instant("2026-03-09T04:00:00Z")
    );
    assert_eq!(
        advance_zoned(anchor, &Frequency::Weekly, 1, &ny()).unwrap(),
        instant("2026-03-14T04:00:00Z")
    );
    let friday = instant("2026-03-06T05:00:00Z");
    assert_eq!(
        advance_zoned(friday, &Frequency::BusinessDaily, 1, &ny()).unwrap(),
        instant("2026-03-09T04:00:00Z")
    );
    let before_gap = instant("2026-03-08T06:30:00Z");
    assert_eq!(
        advance_zoned(before_gap, &Frequency::Hourly, 1, &ny()).unwrap(),
        instant("2026-03-08T07:30:00Z")
    );
}

#[test]
fn ambiguous_and_nonexistent_local_calendar_times_return_errors() {
    // 02:30 EST on March 7 would become the missing 02:30 on March 8.
    let before_gap = instant("2026-03-07T07:30:00Z");
    assert!(
        matches!(advance_zoned(before_gap, &Frequency::Daily, 1, &ny()),
        Err(Error::DataError(message)) if message.contains("nonexistent"))
    );
    // 01:30 EDT on October 31 would become the repeated 01:30 on November 1.
    let before_repeat = instant("2026-10-31T05:30:00Z");
    assert!(
        matches!(advance_zoned(before_repeat, &Frequency::Daily, 1, &ny()),
        Err(Error::DataError(message)) if message.contains("ambiguous"))
    );
    let unknown = DataType::Datetime(TimeUnit::Milliseconds, Some("Unknown/Timezone".into()));
    assert!(advance_zoned(before_gap, &Frequency::Daily, 1, &unknown).is_err());
}

#[test]
fn timestamp_columns_preserve_timezones_and_all_units() {
    let values = [instant("2026-03-31T04:00:00Z")];
    for unit in [
        TimeUnit::Milliseconds,
        TimeUnit::Microseconds,
        TimeUnit::Nanoseconds,
    ] {
        let dtype = DataType::Datetime(unit, Some("America/New_York".into()));
        let column = timestamp_column("ds", &values, &dtype).unwrap();
        assert_eq!(column.dtype(), &dtype);
        assert_eq!(timestamps(&column).unwrap(), values);
    }
}

#[test]
fn calendar_anchors_use_the_local_date_even_when_utc_date_differs() {
    let tokyo = DataType::Datetime(TimeUnit::Milliseconds, Some("Asia/Tokyo".into()));
    let january_midnight = instant("2026-01-30T15:00:00Z");
    assert_eq!(
        advance_zoned(january_midnight, &Frequency::Monthly, 1, &tokyo).unwrap(),
        instant("2026-02-27T15:00:00Z")
    );
    assert_eq!(
        advance_zoned(january_midnight, &Frequency::Monthly, 2, &tokyo).unwrap(),
        instant("2026-03-30T15:00:00Z")
    );
}
